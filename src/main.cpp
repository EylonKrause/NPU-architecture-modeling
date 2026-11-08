#include "npu/core.hpp"
#include "npu/model.hpp"
#include "npu/options.hpp"
#include "npu/paths.hpp"
#include "npu/version.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
// Result and snapshot JSON layout. 2: "config" nested by option-table path
// (shape.m, arch.frequency_mhz, ...). 1: flat config keys (m, frequency_mhz, ...).
constexpr int schema_version = 2;

void diagnostic_report(const sc_core::sc_report& report, const sc_core::sc_actions& actions) {
    if (actions & sc_core::SC_DISPLAY) std::cerr << report.what() << '\n';
    sc_core::sc_report_handler::default_handler(
        report, actions & ~static_cast<sc_core::sc_actions>(sc_core::SC_DISPLAY));
}

struct Arguments {
    npu::ConfigBuilder builder;
    std::vector<std::string> config_paths;
    std::string input_path;
    std::string quantization_path;
    std::string vcd_path, progress_path;
    std::uint64_t slice_cycles = 1024;
    bool live_stdio = false;
    bool dump = false, describe = false;
};

std::string nonempty_path(std::string_view flag, std::string_view value) {
    if (value.empty())
        throw std::invalid_argument(std::string(flag) + ": expected a nonempty path");
    return std::string(value);
}

// Run-control flags choose inputs, outputs and the session mode; results do
// not echo them. A flag with an empty placeholder takes no value.
struct RunOption {
    std::string_view flag, placeholder, help;
    void (*apply)(Arguments& args, std::string_view value);
};

const RunOption run_options[] = {
    {"--config", "FILE.json", "load config-file keys; later flags and files override them",
     [](Arguments& a, std::string_view v) {
         a.config_paths.push_back(nonempty_path("--config", v));
         a.builder.load_file(a.config_paths.back());
     }},
    {"--input", "PATH",
     "NPU1 tensor file: M N K, then activations (M x G*K), weights (G blocks of K x N) and "
     "bias (G*N); G is --groups",
     [](Arguments& a, std::string_view v) { a.input_path = nonempty_path("--input", v); }},
    {"--quantization-file", "PATH",
     "NPUQ1 file: G*N, then weight_zero multiplier shift per output channel; signed shifts",
     [](Arguments& a, std::string_view v) {
         a.quantization_path = nonempty_path("--quantization-file", v);
     }},
    {"--vcd", "PATH.vcd", "write SystemC waveforms",
     [](Arguments& a, std::string_view v) {
         if (v.empty() || std::filesystem::path(v).extension() != ".vcd")
             throw std::invalid_argument("--vcd: expected a nonempty .vcd path");
         a.vcd_path = std::string(v);
     }},
    {"--progress-jsonl", "PATH", "write a snapshot after every slice of simulated cycles",
     [](Arguments& a, std::string_view v) {
         a.progress_path = nonempty_path("--progress-jsonl", v);
     }},
    {"--slice-cycles", "N", "progress slice length (default 1024)",
     [](Arguments& a, std::string_view v) {
         a.slice_cycles = npu::parse_uint64(v, "--slice-cycles");
     }},
    {"--live-stdio", "", "read advance N, snapshot and stop on stdin; answer in JSON lines",
     [](Arguments& a, std::string_view) { a.live_stdio = true; }},
    {"--dump-output", "", "include accumulators and outputs in the result",
     [](Arguments& a, std::string_view) { a.dump = true; }},
    {"--describe-config", "", "print the effective configuration as JSON and exit",
     [](Arguments& a, std::string_view) { a.describe = true; }},
    {"--version", "", "print the version and exit", [](Arguments&, std::string_view) {}},
    {"--help", "", "print this text and exit", [](Arguments&, std::string_view) {}},
};

const RunOption* find_run_option(std::string_view flag) {
    for (const auto& option : run_options)
        if (option.flag == flag) return &option;
    return nullptr;
}

void usage() {
    std::cout << "npu-sim: SystemC model of a tiled INT8 GEMM accelerator\n"
                 "Usage: npu-sim [options]\n\n"
                 "Run control:\n";
    for (const auto& option : run_options)
        npu::write_help_row(std::cout, option.flag, option.placeholder, "", option.help);
    std::cout << "\nConfiguration (flag, config-file key). Flags and --config files apply\n"
                 "left to right; a later assignment replaces an earlier one. Keys\n"
                 "without a flag are set from a config file.\n";
    npu::write_config_help(std::cout);
}

void version() { std::cout << "npu-sim " << npu::version << " (" << npu::git_describe << ")\n"; }

Arguments parse(int argc, char* argv[]) {
    Arguments args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag(argv[i]);
        const auto value = [&]() -> std::string_view {
            if (i + 1 == argc)
                throw std::invalid_argument("missing value for " + std::string(flag));
            return argv[++i];
        };
        if (const auto* run = find_run_option(flag)) {
            run->apply(args, run->placeholder.empty() ? std::string_view{} : value());
        } else if (const auto* option = npu::find_option_by_flag(flag)) {
            if (npu::is_switch(*option)) args.builder.set_switch(*option);
            else args.builder.set(*option, value());
        } else {
            throw std::invalid_argument("unknown option: " + std::string(flag));
        }
    }
    const auto& compute = args.builder.config.arch.compute;
    if (args.builder.source("arch.compute.macs_per_cycle") == npu::OptionSource::CommandLine &&
        (compute.array_rows || compute.array_cols))
        throw std::invalid_argument(
            "choose --macs-per-cycle or --array-rows/--array-cols, not both");
    // An echoed vector-mode config carries dataflow os with no array; the
    // flag applies to an array only.
    if (args.builder.source("arch.compute.dataflow") == npu::OptionSource::CommandLine &&
        !compute.array_rows && !compute.array_cols)
        throw std::invalid_argument("--dataflow requires --array-rows and --array-cols");
    // Each memory model reads its own keys; a flag for the other model would
    // change nothing. Config files may carry both, as every echo does.
    const bool axi = args.builder.config.arch.memory.model == npu::MemoryModel::Axi;
    for (const auto& option : npu::config_options()) {
        if (option.flag.empty() || option.path.substr(0, 12) != "arch.memory." ||
            option.path == "arch.memory.model" ||
            args.builder.source(option.path) != npu::OptionSource::CommandLine)
            continue;
        const bool axi_key = option.path == "arch.memory.dram_bytes_per_cycle" ||
                             option.path.substr(0, 16) == "arch.memory.axi.";
        if (axi_key && !axi)
            throw std::invalid_argument(std::string(option.flag) + " requires --memory-model axi");
        if (!axi_key && axi)
            throw std::invalid_argument(std::string(option.flag) +
                                        " applies to --memory-model serial only");
    }
    if (args.slice_cycles == 0 || args.slice_cycles > 10000000)
        throw std::invalid_argument("slice cycles must be in 1..10000000");
    if (args.live_stdio && (args.dump || !args.builder.config.sim.trace_path.empty()))
        throw std::invalid_argument(
            "--live-stdio reports through snapshots; --trace and --dump-output need a batch run");
    if (args.builder.config.sim.timing_only && (args.dump || !args.input_path.empty()))
        throw std::invalid_argument(
            "--timing-only runs without tensors; --input and --dump-output need a bit-exact run");
    const auto& quant = args.builder.config.quant;
    if (!args.quantization_path.empty() &&
        (!quant.weight_zero_points.empty() || !quant.multipliers.empty() || !quant.shifts.empty()))
        throw std::invalid_argument(
            "per-channel parameters come from --quantization-file or a config file, not both");
    return args;
}

void validate_paths(const Arguments& args) {
    const std::vector<std::pair<std::string, std::string>> outputs = {
        {"VCD", args.vcd_path},
        {"progress JSONL", args.progress_path},
        {"CSV trace", args.builder.config.sim.trace_path}};
    auto inputs = args.config_paths;
    inputs.push_back(args.input_path);
    inputs.push_back(args.quantization_path);
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        if (outputs[i].second.empty()) continue;
        for (std::size_t j = i + 1; j < outputs.size(); ++j) {
            if (!outputs[j].second.empty() &&
                npu::same_file_path(outputs[i].second, outputs[j].second))
                throw std::invalid_argument("output paths conflict: " + outputs[i].first + " and " +
                                            outputs[j].first);
        }
        for (const auto& input : inputs) {
            if (!input.empty() && npu::same_file_path(outputs[i].second, input))
                throw std::invalid_argument("output path conflicts with an input file: " +
                                            outputs[i].first);
        }
    }
}

bool input_token(std::istream& stream, std::string& token) {
    token.clear();
    stream >> std::ws;
    if (!stream.good()) return false;
    while (true) {
        const auto next = stream.peek();
        if (next == std::char_traits<char>::eof() || std::isspace(static_cast<unsigned char>(next)))
            break;
        if (token.size() == 32) throw std::invalid_argument("input token exceeds 32 characters");
        token.push_back(static_cast<char>(stream.get()));
    }
    return !token.empty();
}

npu::Problem input_problem(Arguments& args) {
    if (args.builder.config.sim.timing_only) {
        // The model reads only the shape; no tensors exist.
        const auto& shape = args.builder.config.shape;
        npu::Problem shell;
        shell.m = shape.m;
        shell.n = shape.n;
        shell.k = shape.k;
        shell.groups = shape.groups;
        return shell;
    }
    if (args.input_path.empty()) return npu::make_problem(args.builder.config);
    if (npu::implicit_conv(args.builder.config.shape))
        throw std::invalid_argument(
            "--input holds a GEMM operand; with shape.conv the input tensor is generated");
    const char* const dimensions[] = {"shape.m", "shape.n", "shape.k"};
    for (const auto* path : dimensions)
        if (args.builder.source(path) == npu::OptionSource::CommandLine)
            throw std::invalid_argument(
                "--input dimensions come from its header; omit --m/--n/--k");
    std::ifstream file(args.input_path);
    if (!file) throw std::invalid_argument("cannot open input tensor file");
    std::string magic, header[3];
    if (!input_token(file, magic) || !input_token(file, header[0]) ||
        !input_token(file, header[1]) || !input_token(file, header[2]) || magic != "NPU1")
        throw std::invalid_argument("input header must be NPU1 M N K");
    auto& c = args.builder.config;
    std::uint64_t* const fields[] = {&c.shape.m, &c.shape.n, &c.shape.k};
    for (std::size_t i = 0; i < 3; ++i) {
        const auto value = npu::parse_uint64(header[i], "NPU1 header");
        // A config file may restate the shape, as a result's echoed config
        // does, but it must agree with the tensors.
        if (args.builder.source(dimensions[i]) == npu::OptionSource::File && *fields[i] != value)
            throw std::invalid_argument(std::string("--input header gives ") + dimensions[i] +
                                        " = " + header[i] + " but the config file sets " +
                                        std::to_string(*fields[i]));
        *fields[i] = value;
    }
    npu::validate(c);
    npu::Problem p;
    p.m = c.shape.m;
    p.n = c.shape.n;
    p.k = c.shape.k;
    p.groups = c.shape.groups;
    const auto channels = npu::checked_mul(c.shape.groups, c.shape.n);
    p.activations.resize(static_cast<std::size_t>(
        npu::checked_mul(c.shape.m, npu::checked_mul(c.shape.groups, c.shape.k))));
    p.weights.resize(static_cast<std::size_t>(npu::checked_mul(c.shape.k, channels)));
    p.bias.resize(static_cast<std::size_t>(channels));
    for (auto* values : {&p.activations, &p.weights}) {
        for (auto& value : *values) {
            std::string token;
            if (!input_token(file, token)) throw std::invalid_argument("truncated INT8 tensor");
            const auto x = npu::parse_int32(token, "NPU1 tensor");
            const bool weight = values == &p.weights;
            const auto low = weight && c.weights.bits == 4 ? -8 : -128;
            const auto high = weight && c.weights.bits == 4 ? 7 : 127;
            if (x < low || x > high)
                throw std::invalid_argument("tensor value outside configured signed precision");
            value = static_cast<std::int8_t>(x);
        }
    }
    for (auto& value : p.bias) {
        std::string token;
        if (!input_token(file, token)) throw std::invalid_argument("truncated bias tensor");
        value = npu::parse_int32(token, "NPU1 bias");
    }
    std::string extra;
    if (input_token(file, extra)) throw std::invalid_argument("unexpected trailing input token");
    if (!file.eof()) throw std::invalid_argument("input tensor read failed");
    return p;
}

void input_quantization(Arguments& args) {
    if (args.quantization_path.empty()) return;
    std::ifstream file(args.quantization_path);
    if (!file) throw std::invalid_argument("cannot open quantization parameter file");
    std::string magic, count;
    const auto& shape = args.builder.config.shape;
    const auto channels = npu::checked_mul(shape.groups, shape.n);
    if (!input_token(file, magic) || !input_token(file, count) || magic != "NPUQ1" ||
        npu::parse_uint64(count, "NPUQ1 header") != channels)
        throw std::invalid_argument(
            "quantization header must be NPUQ1 G*N matching output channels");
    auto& q = args.builder.config.quant;
    for (std::uint64_t channel = 0; channel < channels; ++channel) {
        std::string zero, multiplier, shift;
        if (!input_token(file, zero) || !input_token(file, multiplier) || !input_token(file, shift))
            throw std::invalid_argument("truncated quantization parameters");
        q.weight_zero_points.push_back(npu::parse_int32(zero, "NPUQ1 weight zero point"));
        q.multipliers.push_back(npu::parse_int32(multiplier, "NPUQ1 multiplier"));
        q.shifts.push_back(npu::parse_int32(shift, "NPUQ1 shift"));
    }
    std::string extra;
    if (input_token(file, extra))
        throw std::invalid_argument("unexpected trailing quantization token");
    if (!file.eof()) throw std::invalid_argument("quantization read failed");
}

template <class T>
void array_json(const std::vector<T>& values, std::ostream& stream = std::cout) {
    stream << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) stream << ',';
        stream << static_cast<std::int64_t>(values[i]);
    }
    stream << ']';
}

const char* numerics_name(const npu::Config& c) {
    return c.sim.timing_only ? "skipped" : "bit-exact";
}

void check_numerics(const npu::Config& c, const npu::Accelerator& model,
                    const npu::Output& expected) {
    if (c.sim.timing_only) return;
    if (model.completed() && (model.output().accumulators != expected.accumulators ||
                              model.output().quantized != expected.quantized))
        throw std::runtime_error("numerical validation failed");
}

double fraction(std::uint64_t value, std::uint64_t total) {
    return total ? static_cast<double>(value) / static_cast<double>(total) : 0.0;
}

// Per channel: counters, request latency, burst efficiency = payload /
// (beats * beat bytes) and payload bytes per cycle of the run; snapshots add
// the channel state. Beat bytes are the serial bytes_per_cycle or the AXI
// data_bytes.
void channels_json(std::ostream& stream, const npu::Config& c, const npu::MemoryView& memory,
                   std::uint64_t cycles, bool live) {
    const auto& m = c.arch.memory;
    const auto beat_bytes = m.model == npu::MemoryModel::Axi ? m.axi.data_bytes : m.bytes_per_cycle;
    const auto channel = [&](const char* name, const npu::ChannelView& view) {
        const auto& s = view.statistics;
        stream << '"' << name << "\":{\"requests\":" << s.requests << ",\"bursts\":" << s.bursts
               << ",\"beats\":" << s.beats << ",\"payload_bytes\":" << s.payload_bytes
               << ",\"busy_cycles\":" << s.busy_cycles
               << ",\"mean_latency_cycles\":" << fraction(s.latency_cycles, s.requests)
               << ",\"max_latency_cycles\":" << s.max_latency_cycles << ",\"burst_efficiency\":"
               << fraction(s.payload_bytes, npu::checked_mul(s.beats, beat_bytes))
               << ",\"achieved_bytes_per_cycle\":" << fraction(s.payload_bytes, cycles)
               << ",\"peak_outstanding\":" << s.peak_outstanding;
        if (live)
            stream << ",\"active\":" << view.active << ",\"outstanding\":" << view.outstanding
                   << ",\"queued\":" << view.queued;
        stream << '}';
    };
    channel("read", memory.read);
    stream << ',';
    channel("write", memory.write);
}

// The cluster's memory: every port combined, and under axi the shared DRAM.
void memory_json(std::ostream& stream, const npu::Config& c, const npu::MemoryView& memory,
                 std::uint64_t cycles, bool live) {
    const auto& m = c.arch.memory;
    const bool axi = m.model == npu::MemoryModel::Axi;
    stream << "{\"model\":\"" << npu::to_string(m.model) << '"'
           << ",\"peak_bytes_per_cycle\":" << npu::peak_memory_bytes_per_cycle(c) << ',';
    channels_json(stream, c, memory, cycles, live);
    if (axi)
        stream << ",\"dram\":{\"bytes_per_cycle\":" << m.dram_bytes_per_cycle
               << ",\"bytes\":" << memory.dram_bytes << ",\"utilization\":"
               << fraction(memory.dram_bytes, npu::checked_mul(m.dram_bytes_per_cycle, cycles))
               << '}';
    stream << '}';
}

// The plan's static fields; the caller closes the object.
void plan_json(std::ostream& stream, const npu::ClusterPlan& plan) {
    stream << "{\"cores\":" << plan.cores << ",\"partition\":\"" << npu::to_string(plan.partition)
           << "\",\"split\":\"" << npu::to_string(plan.split) << "\",\"units\":" << plan.units;
}

void snapshot_json(std::ostream& stream, const npu::Config& c, const npu::Problem& problem,
                   const npu::Accelerator& model, const char* type = "snapshot") {
    const auto snap = model.snapshot();
    const auto& s = snap.statistics;
    const auto tiles = npu::tile_count(c);
    const auto blocks = npu::reduction_blocks(c);
    // A timing-only run has nothing to validate or hash.
    const bool checked = snap.completed && !c.sim.timing_only;
    // Cluster utilizations are the mean over cores.
    const auto core_cycles = npu::checked_mul(snap.cycles, model.cores());
    const auto ratio = [core_cycles](std::uint64_t busy) { return fraction(busy, core_cycles); };
    stream << std::boolalpha << std::setprecision(17) << "{\"schema_version\":" << schema_version
           << ",\"type\":\"" << type << "\",\"simulation\":true,\"config\":";
    npu::write_config_json(stream, c);
    stream << ",\"cycles\":" << snap.cycles << ",\"completed\":" << snap.completed
           << ",\"validated\":" << checked << ",\"numerics\":\"" << numerics_name(c) << '"'
           << ",\"numerical_match\":" << (checked ? "true" : "null") << ",\"macs\":" << s.macs
           << ",\"read_bytes\":" << s.read_bytes << ",\"write_bytes\":" << s.write_bytes
           << ",\"executed_macs\":" << s.executed_macs << ",\"skipped_macs\":" << s.skipped_macs
           << ",\"weight_decode_cycles\":" << s.weight_decode_cycles
           << ",\"decoded_weight_values\":" << s.decoded_weight_values
           << ",\"weight_read_bytes\":" << s.weight_read_bytes
           << ",\"encoded_weight_storage_bytes\":" << s.encoded_weight_storage_bytes
           << ",\"dense_weight_storage_bytes\":" << s.dense_weight_storage_bytes
           << ",\"compute_busy_cycles\":" << s.compute_busy_cycles
           << ",\"dma_busy_cycles\":" << s.dma_busy_cycles
           << ",\"compute_utilization\":" << ratio(s.compute_busy_cycles)
           << ",\"memory_utilization\":" << ratio(s.dma_busy_cycles)
           << ",\"inflight_tiles\":" << snap.inflight_tiles
           << ",\"read_queue_depth\":" << snap.read_queue_depth
           << ",\"write_queue_depth\":" << snap.write_queue_depth
           << ",\"dma_busy\":" << snap.dma_busy << ",\"compute_busy\":" << snap.compute_busy
           << ",\"dma_read\":" << snap.dma_read << ",\"dma_write\":" << snap.dma_write
           << ",\"dma_tile_id\":" << snap.dma_tile_id << ",\"dma_block_id\":" << snap.dma_block_id
           << ",\"compute_tile_id\":" << snap.compute_tile_id
           << ",\"compute_block_id\":" << snap.compute_block_id << ",\"memory\":";
    memory_json(stream, c, snap.memory, snap.cycles, true);
    stream << ",\"cluster\":";
    plan_json(stream, model.plan());
    stream << "},\"cores\":[";
    for (std::uint64_t core = 0; core < model.cores(); ++core) {
        const auto& view = snap.cores[static_cast<std::size_t>(core)];
        const auto& v = view.statistics;
        const auto core_tiles = model.core_tiles(core);
        stream << (core ? "," : "") << "{\"core\":" << core << ",\"completed\":" << view.completed
               << ",\"cycles\":" << view.cycles << ",\"tiles_total\":" << core_tiles
               << ",\"tiles_completed\":" << v.completed_tiles
               << ",\"panels_total\":" << npu::checked_mul(core_tiles, blocks)
               << ",\"panels_completed\":" << v.computed_panels
               << ",\"loaded_panels\":" << v.loaded_panels
               << ",\"inflight_tiles\":" << view.inflight_tiles << ",\"dma_busy\":" << view.dma_busy
               << ",\"compute_busy\":" << view.compute_busy << ",\"dma_read\":" << view.dma_read
               << ",\"dma_write\":" << view.dma_write << ",\"dma_tile_id\":" << view.dma_tile_id
               << ",\"dma_block_id\":" << view.dma_block_id
               << ",\"compute_tile_id\":" << view.compute_tile_id
               << ",\"compute_block_id\":" << view.compute_block_id << ",\"macs\":" << v.macs
               << ",\"read_bytes\":" << v.read_bytes << ",\"write_bytes\":" << v.write_bytes
               << ",\"compute_busy_cycles\":" << v.compute_busy_cycles
               << ",\"dma_busy_cycles\":" << v.dma_busy_cycles << ",\"memory\":{";
        channels_json(stream, c, view.memory, snap.cycles, true);
        stream << "}}";
    }
    stream << "],\"tiles_total\":" << tiles << ",\"tiles_completed\":" << s.completed_tiles
           << ",\"panels_total\":" << npu::checked_mul(tiles, blocks)
           << ",\"panels_completed\":" << s.computed_panels
           << ",\"loaded_panels\":" << s.loaded_panels << ",\"modeled_latency_ns\":"
           << static_cast<double>(snap.cycles) * 1000.0 / c.arch.frequency_mhz
           << ",\"active_stages\":[";
    const char* separator = "";
    for (const auto& [active, stage] :
         {std::pair{snap.memory.read.active, "load"}, std::pair{snap.memory.write.active, "store"},
          std::pair{snap.compute_busy, "compute"}}) {
        if (!active) continue;
        stream << separator << '"' << stage << '"';
        separator = ",";
    }
    stream << "],\"output_hash\":";
    if (checked) stream << '"' << npu::output_hash(model.output()) << '"';
    else stream << "null";
    stream << ",\"input_preview\":";
    array_json(std::vector<std::int8_t>(
                   problem.activations.begin(),
                   problem.activations.begin() + static_cast<std::ptrdiff_t>(std::min<std::size_t>(
                                                     16, problem.activations.size()))),
               stream);
    stream << ",\"output_preview\":";
    if (snap.completed)
        array_json(std::vector<std::int8_t>(model.output().quantized.begin(),
                                            model.output().quantized.begin() +
                                                static_cast<std::ptrdiff_t>(std::min<std::size_t>(
                                                    16, model.output().quantized.size()))),
                   stream);
    else stream << "[]";
    stream << "}\n" << std::flush;
    if (!stream) throw std::runtime_error("cannot write simulation progress");
}

void live_commands(const npu::Config& c, const npu::Problem& problem, npu::Accelerator& model,
                   const npu::Output& expected, std::ostream* progress) {
    sc_core::sc_start(sc_core::SC_ZERO_TIME);
    const auto emit = [&](const char* type = "snapshot") {
        check_numerics(c, model, expected);
        snapshot_json(std::cout, c, problem, model, type);
        if (progress) snapshot_json(*progress, c, problem, model, type);
    };
    emit();
    while (true) {
        char command[257] = {};
        if (!std::cin.getline(command, sizeof(command))) {
            if (std::cin.eof()) return;
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "{\"type\":\"error\",\"error\":\"command exceeds 256 characters\"}\n"
                      << std::flush;
            continue;
        }
        std::istringstream line(command);
        std::string verb, value, extra;
        line >> verb;
        if (verb == "advance" && (line >> value) && !(line >> extra)) {
            try {
                const auto cycles = npu::parse_uint64(value, "advance");
                if (!cycles || cycles > 10000000) throw std::invalid_argument("invalid slice");
                if (!model.completed()) {
                    const auto before = sc_core::sc_time_stamp();
                    sc_core::sc_start(sc_core::sc_time::from_value(cycles),
                                      sc_core::SC_RUN_TO_TIME);
                    if (!model.completed() &&
                        (sc_core::sc_time_stamp() == before || !sc_core::sc_pending_activity()))
                        throw std::runtime_error("simulation stalled before completion");
                }
            } catch (const std::invalid_argument&) {
                std::cout
                    << "{\"type\":\"error\",\"error\":\"advance requires cycles in 1..10000000\"}\n"
                    << std::flush;
                continue;
            }
            emit();
        } else if ((verb == "snapshot" || verb == "stop") && !(line >> extra)) {
            emit(verb == "stop" ? "stopped" : "snapshot");
            if (verb == "stop") return;
        } else {
            std::cout
                << "{\"type\":\"error\",\"error\":\"expected advance N, snapshot, or stop\"}\n"
                << std::flush;
        }
    }
}

// Speedup inputs: the makespan, the MAC imbalance (largest core over the mean)
// and the DRAM floor, then both splits' predictions and every core's counters.
void cluster_json(std::ostream& stream, const npu::Config& c, const npu::Accelerator& model,
                  const npu::Statistics& total) {
    const auto& memory = c.arch.memory;
    const bool axi = memory.model == npu::MemoryModel::Axi;
    const auto dram = model.memory().dram_bytes;
    std::uint64_t largest = 0;
    for (std::uint64_t core = 0; core < model.cores(); ++core)
        largest = std::max(largest, model.statistics(core).macs);
    stream << ",\n\"cluster\":";
    plan_json(stream, model.plan());
    stream << ",\"makespan_cycles\":" << total.cycles
           << ",\"imbalance\":" << fraction(npu::checked_mul(largest, model.cores()), total.macs)
           << ",\"dram_bound_cycles\":"
           << (axi && memory.dram_bytes_per_cycle ?
                   npu::ceil_div(dram, memory.dram_bytes_per_cycle) :
                   0)
           << ",\"predicted\":{";
    for (const auto partition : {npu::Partition::M, npu::Partition::N}) {
        const auto& estimate = model.estimate(partition);
        stream << (partition == npu::Partition::M ? "" : ",") << '"' << npu::to_string(partition)
               << "\":{\"read_bytes\":" << estimate.read_bytes
               << ",\"write_bytes\":" << estimate.write_bytes
               << ",\"bound_cycles\":" << estimate.bound_cycles << '}';
    }
    stream << "}},\n\"cores\":[";
    const auto& plan = model.plan();
    for (std::uint64_t core = 0; core < model.cores(); ++core) {
        const auto s = model.statistics(core);
        stream << (core ? ",\n" : "\n") << "{\"core\":" << core
               << ",\"first_unit\":" << plan.first_unit(core)
               << ",\"units\":" << plan.unit_count(core) << ",\"tiles\":" << model.core_tiles(core)
               << ",\"cycles\":" << s.cycles << ",\"macs\":" << s.macs
               << ",\"executed_macs\":" << s.executed_macs
               << ",\"compute_busy_cycles\":" << s.compute_busy_cycles
               << ",\"dma_busy_cycles\":" << s.dma_busy_cycles
               << ",\"compute_utilization\":" << fraction(s.compute_busy_cycles, total.cycles)
               << ",\"memory_utilization\":" << fraction(s.dma_busy_cycles, total.cycles)
               << ",\"slot_wait_cycles\":" << s.slot_wait_cycles
               << ",\"panel_wait_cycles\":" << s.panel_wait_cycles
               << ",\"peak_inflight_tiles\":" << s.peak_inflight_tiles
               << ",\"read_bytes\":" << s.read_bytes << ",\"write_bytes\":" << s.write_bytes
               << ",\"weight_read_bytes\":" << s.weight_read_bytes
               << ",\"weight_cache_hits\":" << s.weight_cache_hits
               << ",\"weight_cache_misses\":" << s.weight_cache_misses
               << ",\"activation_read_bytes\":" << s.activation_read_bytes
               << ",\"activation_cache_hits\":" << s.activation_cache_hits
               << ",\"activation_cache_misses\":" << s.activation_cache_misses
               << ",\"depthwise_macs\":" << s.depthwise_macs << ",\"memory\":{";
        const auto port = model.memory(core);
        channels_json(stream, c, port, total.cycles, false);
        if (axi) stream << ",\"dram_bytes\":" << port.dram_bytes;
        stream << "}}";
    }
    stream << "\n]";
}

void report(const npu::Config& c, const npu::Accelerator& model, double seconds, bool dump) {
    const auto s = model.statistics();
    const auto cores = model.cores();
    // Utilizations are the mean over cores; the peak rates are the cluster's.
    const auto core_cycles = static_cast<double>(npu::checked_mul(s.cycles, cores));
    // The peak of the datapath that runs the problem; the depthwise datapath's
    // rate and MACs are also reported on their own.
    const auto peak_macs = npu::checked_mul(npu::peak_macs_per_cycle(c), cores);
    const auto depthwise_peak = npu::checked_mul(c.arch.compute.depthwise_macs_per_cycle, cores);
    // Each core's busiest resource, and the DRAM floor all cores share.
    std::uint64_t lower_bound = 0;
    for (std::uint64_t core = 0; core < cores; ++core) {
        const auto each = model.statistics(core);
        lower_bound = std::max({lower_bound, each.compute_busy_cycles, each.dma_busy_cycles});
    }
    const auto& memory = c.arch.memory;
    if (memory.model == npu::MemoryModel::Axi && memory.dram_bytes_per_cycle)
        lower_bound = std::max(
            lower_bound, npu::ceil_div(model.memory().dram_bytes, memory.dram_bytes_per_cycle));
    std::cout << std::setprecision(17) << std::boolalpha;
    std::cout << "{\n\"schema_version\":" << schema_version << ",\n\"config\":";
    npu::write_config_json(std::cout, c);
    std::cout << ",\n"
              << "\"cycles\":" << s.cycles << ",\n\"macs\":" << s.macs
              << ",\n\"executed_macs\":" << s.executed_macs
              << ",\n\"skipped_macs\":" << s.skipped_macs
              << ",\n\"weight_decode_cycles\":" << s.weight_decode_cycles
              << ",\n\"decoded_weight_values\":" << s.decoded_weight_values
              << ",\n\"weight_read_bytes\":" << s.weight_read_bytes
              << ",\n\"encoded_weight_storage_bytes\":" << s.encoded_weight_storage_bytes
              << ",\n\"dense_weight_storage_bytes\":" << s.dense_weight_storage_bytes
              << ",\n\"read_bytes\":" << s.read_bytes << ",\n\"write_bytes\":" << s.write_bytes
              << ",\n\"compute_busy_cycles\":" << s.compute_busy_cycles
              << ",\n\"dma_busy_cycles\":" << s.dma_busy_cycles
              << ",\n\"peak_inflight_tiles\":" << s.peak_inflight_tiles
              << ",\n\"slot_wait_cycles\":" << s.slot_wait_cycles
              << ",\n\"panel_wait_cycles\":" << s.panel_wait_cycles
              << ",\n\"reduction_blocks\":" << npu::reduction_blocks(c)
              << ",\n\"effective_macs_per_cycle\":" << peak_macs
              << ",\n\"peak_memory_bytes_per_cycle\":" << npu::peak_memory_bytes_per_cycle(c)
              << ",\n\"dma_requests\":" << s.dma_requests << ",\n\"dma_bursts\":" << s.dma_bursts
              << ",\n\"weight_cache_hits\":" << s.weight_cache_hits
              << ",\n\"weight_cache_misses\":" << s.weight_cache_misses
              << ",\n\"weight_bytes_saved\":" << s.weight_bytes_saved
              << ",\n\"peak_cached_weight_bytes\":" << s.peak_cached_weight_bytes
              << ",\n\"activation_read_bytes\":" << s.activation_read_bytes
              << ",\n\"activation_cache_hits\":" << s.activation_cache_hits
              << ",\n\"activation_cache_misses\":" << s.activation_cache_misses
              << ",\n\"activation_bytes_saved\":" << s.activation_bytes_saved
              << ",\n\"peak_cached_activation_bytes\":" << s.peak_cached_activation_bytes
              << ",\n\"depthwise_macs\":" << s.depthwise_macs
              << ",\n\"depthwise_peak_macs_per_cycle\":" << depthwise_peak
              << ",\n\"depthwise_mac_utilization\":"
              << fraction(s.depthwise_macs, npu::checked_mul(depthwise_peak, s.cycles))
              << ",\n\"scratchpad_bytes\":" << npu::scratchpad_bytes(c) << ",\n\"numerics\":\""
              << numerics_name(c) << '"';
    if (c.sim.timing_only) std::cout << ",\n\"output_hash\":null,\n\"numerical_match\":null";
    else
        std::cout << ",\n\"output_hash\":" << npu::output_hash(model.output())
                  << ",\n\"numerical_match\":true";
    std::cout << ",\n\"simulation_wall_seconds\":" << seconds << ",\n\"compute_utilization\":"
              << static_cast<double>(s.compute_busy_cycles) / core_cycles
              << ",\n\"memory_utilization\":"
              << static_cast<double>(s.dma_busy_cycles) / core_cycles << ",\n\"mac_utilization\":"
              << static_cast<double>(s.executed_macs) /
                     (static_cast<double>(peak_macs) * static_cast<double>(s.cycles))
              << ",\n\"dma_payload_utilization\":"
              << static_cast<double>(s.read_bytes + s.write_bytes) /
                     (static_cast<double>(npu::peak_memory_bytes_per_cycle(c)) *
                      static_cast<double>(s.cycles))
              << ",\n\"arithmetic_intensity_macs_per_byte\":"
              << static_cast<double>(s.macs) / static_cast<double>(s.read_bytes + s.write_bytes)
              << ",\n\"resource_lower_bound_cycles\":" << lower_bound
              << ",\n\"modeled_latency_ns\":"
              << static_cast<double>(s.cycles) * 1000.0 / c.arch.frequency_mhz << ",\n\"memory\":";
    memory_json(std::cout, c, model.memory(), s.cycles, false);
    cluster_json(std::cout, c, model, s);
    if (dump) {
        std::cout << ",\n\"accumulators\":";
        array_json(model.output().accumulators);
        std::cout << ",\n\"outputs\":";
        array_json(model.output().quantized);
    }
    std::cout << "\n}\n";
}
}  // namespace

int sc_main(int argc, char* argv[]) {
    try {
        // --help and --version answer even when other arguments are invalid.
        for (int i = 1; i < argc; ++i) {
            const std::string_view flag(argv[i]);
            if (flag == "--help") {
                usage();
                return 0;
            }
            if (flag == "--version") {
                version();
                return 0;
            }
        }
        auto args = parse(argc, argv);
        validate_paths(args);
        const auto problem = input_problem(args);
        input_quantization(args);
        const auto& config = args.builder.config;
        npu::validate(config);
        if (args.describe) {
            npu::write_config_json(std::cout, config);
            std::cout << '\n';
            return 0;
        }
        const auto expected =
            config.sim.timing_only ? npu::Output{} : npu::reference(problem, config.quant);
        sc_core::sc_set_time_resolution(1, sc_core::SC_NS);
        sc_core::sc_report_handler::set_handler(diagnostic_report);
        sc_core::sc_report_handler::set_actions(sc_core::SC_INFO, sc_core::SC_DO_NOTHING);
        npu::Accelerator model("npu", config, problem);
        std::unique_ptr<sc_core::sc_trace_file, decltype(&sc_core::sc_close_vcd_trace_file)> waves(
            nullptr, &sc_core::sc_close_vcd_trace_file);
        if (!args.vcd_path.empty()) {
            std::ofstream probe(args.vcd_path);
            if (!probe) throw std::invalid_argument("cannot open VCD waveform path");
            probe.close();
            auto base = std::filesystem::path(args.vcd_path);
            base.replace_extension();
            waves.reset(sc_core::sc_create_vcd_trace_file(base.string().c_str()));
            if (!waves) throw std::runtime_error("cannot create VCD waveform file");
            waves->set_time_unit(1, sc_core::SC_PS);
            sc_core::sc_trace_delta_cycles(waves.get(), true);
            model.trace_signals(waves.get());
        }
        std::ofstream progress;
        if (!args.progress_path.empty()) {
            progress.open(args.progress_path);
            if (!progress) throw std::invalid_argument("cannot open progress JSONL path");
        }
        if (args.live_stdio) {
            live_commands(config, problem, model, expected,
                          progress.is_open() ? &progress : nullptr);
            return 0;
        }
        const auto before = std::chrono::steady_clock::now();
        if (progress.is_open()) {
            sc_core::sc_start(sc_core::SC_ZERO_TIME);
            snapshot_json(progress, config, problem, model);
            while (!model.completed()) {
                const auto previous = sc_core::sc_time_stamp();
                sc_core::sc_start(sc_core::sc_time::from_value(args.slice_cycles),
                                  sc_core::SC_RUN_TO_TIME);
                if (!model.completed() &&
                    (sc_core::sc_time_stamp() == previous || !sc_core::sc_pending_activity()))
                    throw std::runtime_error("simulation stalled before completion");
                check_numerics(config, model, expected);
                snapshot_json(progress, config, problem, model);
            }
        } else sc_core::sc_start();
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - before).count();
        if (!model.completed()) throw std::runtime_error("model deadlocked before completion");
        check_numerics(config, model, expected);
        if (!config.sim.trace_path.empty()) {
            std::ofstream trace(config.sim.trace_path);
            if (!trace) throw std::runtime_error("cannot open trace output");
            npu::write_trace(trace, model.spans());
        }
        report(config, model, seconds, args.dump);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "npu-sim: " << error.what() << '\n';
        return 2;
    }
}
