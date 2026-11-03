#include "npu/core.hpp"
#include "npu/model.hpp"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
std::uint64_t unsigned_value(const std::string& text) {
    std::uint64_t result = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        throw std::invalid_argument("expected an unsigned integer: " + text);
    return result;
}

std::int32_t signed_value(const std::string& text) {
    std::int32_t result = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        throw std::invalid_argument("expected a signed 32-bit integer: " + text);
    return result;
}

std::uint32_t u32_value(const std::string& text) {
    const auto value = unsigned_value(text);
    if (value > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("integer exceeds uint32 range: " + text);
    return static_cast<std::uint32_t>(value);
}

void usage() {
    std::cout << "NPU architecture modeling: quantized dense-layer SystemC simulator\n"
        "Usage: npu-sim [options]\n"
        "  --m N --n N --k N              matrix dimensions (default 16,16,32)\n"
        "  --tile-m N --tile-n N          output tile shape (default 8,8)\n"
        "  --tile-k N                    reduction panel size (0 means full K)\n"
        "  --weight-bits 4|8             signed weight precision and packed traffic\n"
        "  --array-rows N --array-cols N  output-stationary systolic timing\n"
        "  --macs-per-cycle N            compute throughput (default 64)\n"
        "  --memory-bytes-per-cycle N    shared DMA bandwidth (default 16)\n"
        "  --memory-latency N            latency per transfer (default 12)\n"
        "  --dma-burst-bytes N           burst limit; each burst pays latency (0 unlimited)\n"
        "  --weight-cache-bytes N        bounded LRU weight-panel cache (default 0)\n"
        "  --pipeline-cycles N           fill/drain overhead per tile (default 4)\n"
        "  --buffer-slots N              end-to-end tile slots (default 2)\n"
        "  --sram-bytes N                accelerator SRAM capacity (default 262144)\n"
        "  --seed N                      deterministic seed masked to 32 bits (default 1)\n"
        "  --input PATH                  real tensors: NPU1 M N K, then A, W, bias\n"
        "  --frequency-mhz X             target frequency for time reporting (default 200)\n"
        "  --timing event|cycle          jump stages or step each architecture cycle\n"
        "  --input-zero N --weight-zero N --output-zero N    signed INT8 zero points\n"
        "  --multiplier N --shift N      effective rescale = N / 2^shift (default 1,7)\n"
        "  --quantization-file PATH      NPUQ1 N followed by weight_zero,multiplier,shift triples\n"
        "  --rounding away|even          rounding ties (default away from zero)\n"
        "  --relu                        fused quantized ReLU\n"
        "  --trace PATH                  export stage trace CSV\n"
        "  --dump-output                 include both result arrays in JSON\n"
        "  --help                        show this text\n";
}

struct Arguments {
    npu::Config config;
    std::string input_path;
    std::string quantization_path;
    bool dump = false, help = false;
    bool dimensions_given = false;
    bool macs_given = false;
};

Arguments parse(int argc, char* argv[]) {
    Arguments args;
    for (int i = 1; i < argc; ++i) {
        const std::string flag(argv[i]);
        if (flag == "--help") { args.help = true; continue; }
        if (flag == "--relu") { args.config.quant.relu = true; continue; }
        if (flag == "--dump-output") { args.dump = true; continue; }
        if (i + 1 == argc) throw std::invalid_argument("missing value for " + flag);
        const std::string value(argv[++i]);
        auto& c = args.config;
        if (flag == "--m") { c.m = unsigned_value(value); args.dimensions_given = true; }
        else if (flag == "--n") { c.n = unsigned_value(value); args.dimensions_given = true; }
        else if (flag == "--k") { c.k = unsigned_value(value); args.dimensions_given = true; }
        else if (flag == "--tile-m") c.tile_m = unsigned_value(value);
        else if (flag == "--tile-n") c.tile_n = unsigned_value(value);
        else if (flag == "--tile-k") c.tile_k = unsigned_value(value);
        else if (flag == "--weight-bits") c.weight_bits = unsigned_value(value);
        else if (flag == "--array-rows") c.array_rows = unsigned_value(value);
        else if (flag == "--array-cols") c.array_cols = unsigned_value(value);
        else if (flag == "--macs-per-cycle") { c.macs_per_cycle = unsigned_value(value); args.macs_given = true; }
        else if (flag == "--memory-bytes-per-cycle") c.memory_bytes_per_cycle = unsigned_value(value);
        else if (flag == "--memory-latency") c.memory_latency = unsigned_value(value);
        else if (flag == "--dma-burst-bytes") c.dma_burst_bytes = unsigned_value(value);
        else if (flag == "--weight-cache-bytes") c.weight_cache_bytes = unsigned_value(value);
        else if (flag == "--pipeline-cycles") c.pipeline_cycles = unsigned_value(value);
        else if (flag == "--buffer-slots") c.buffer_slots = unsigned_value(value);
        else if (flag == "--sram-bytes") c.sram_bytes = unsigned_value(value);
        else if (flag == "--seed") c.seed = unsigned_value(value);
        else if (flag == "--input-zero") c.quant.input_zero = signed_value(value);
        else if (flag == "--weight-zero") c.quant.weight_zero = signed_value(value);
        else if (flag == "--output-zero") c.quant.output_zero = signed_value(value);
        else if (flag == "--multiplier") c.quant.multiplier = u32_value(value);
        else if (flag == "--shift") c.quant.shift = u32_value(value);
        else if (flag == "--rounding") c.quant.rounding = value;
        else if (flag == "--timing") c.timing = value;
        else if (flag == "--trace") {
            if (value.empty()) throw std::invalid_argument("--trace path must not be empty");
            c.trace_path = value;
        }
        else if (flag == "--input") {
            if (value.empty()) throw std::invalid_argument("--input path must not be empty");
            args.input_path = value;
        }
        else if (flag == "--quantization-file") {
            if (value.empty()) throw std::invalid_argument("--quantization-file path must not be empty");
            args.quantization_path = value;
        }
        else if (flag == "--frequency-mhz") {
            std::size_t used = 0;
            c.frequency_mhz = std::stod(value, &used);
            if (used != value.size()) throw std::invalid_argument("invalid frequency");
        } else throw std::invalid_argument("unknown option: " + flag);
    }
    if (args.macs_given && (args.config.array_rows || args.config.array_cols))
        throw std::invalid_argument("choose --macs-per-cycle or --array-rows/--array-cols, not both");
    return args;
}

bool input_token(std::istream& stream, std::string& token) {
    token.clear();
    stream >> std::ws;
    if (!stream.good()) return false;
    while (true) {
        const auto next = stream.peek();
        if (next == std::char_traits<char>::eof() ||
            std::isspace(static_cast<unsigned char>(next))) break;
        if (token.size() == 32)
            throw std::invalid_argument("input token exceeds 32 characters");
        token.push_back(static_cast<char>(stream.get()));
    }
    return !token.empty();
}

npu::Problem input_problem(Arguments& args) {
    if (args.input_path.empty()) return npu::make_problem(args.config);
    if (args.dimensions_given)
        throw std::invalid_argument("--input dimensions come from its header; omit --m/--n/--k");
    std::ifstream file(args.input_path);
    if (!file) throw std::invalid_argument("cannot open input tensor file");
    std::string magic, m, n, k;
    if (!input_token(file, magic) || !input_token(file, m) || !input_token(file, n) ||
        !input_token(file, k) || magic != "NPU1")
        throw std::invalid_argument("input header must be NPU1 M N K");
    auto& c = args.config;
    c.m = unsigned_value(m); c.n = unsigned_value(n); c.k = unsigned_value(k);
    npu::validate(c);
    npu::Problem p;
    p.m = c.m; p.n = c.n; p.k = c.k;
    p.activations.resize(static_cast<std::size_t>(npu::checked_mul(c.m, c.k)));
    p.weights.resize(static_cast<std::size_t>(npu::checked_mul(c.k, c.n)));
    p.bias.resize(static_cast<std::size_t>(c.n));
    for (auto* values : {&p.activations, &p.weights}) {
        for (auto& value : *values) {
            std::string token;
            if (!input_token(file, token)) throw std::invalid_argument("truncated INT8 tensor");
            const auto x = signed_value(token);
            const bool weight = values == &p.weights;
            const auto low = weight && c.weight_bits == 4 ? -8 : -128;
            const auto high = weight && c.weight_bits == 4 ? 7 : 127;
            if (x < low || x > high) throw std::invalid_argument("tensor value outside configured signed precision");
            value = static_cast<std::int8_t>(x);
        }
    }
    for (auto& value : p.bias) {
        std::string token;
        if (!input_token(file, token)) throw std::invalid_argument("truncated bias tensor");
        value = signed_value(token);
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
    if (!input_token(file, magic) || !input_token(file, count) || magic != "NPUQ1" ||
        unsigned_value(count) != args.config.n)
        throw std::invalid_argument("quantization header must be NPUQ1 N matching output channels");
    auto& q = args.config.quant;
    for (std::uint64_t channel = 0; channel < args.config.n; ++channel) {
        std::string zero, multiplier, shift;
        if (!input_token(file, zero) || !input_token(file, multiplier) || !input_token(file, shift))
            throw std::invalid_argument("truncated quantization parameters");
        q.weight_zero_points.push_back(signed_value(zero));
        q.multipliers.push_back(u32_value(multiplier));
        q.shifts.push_back(u32_value(shift));
    }
    std::string extra;
    if (input_token(file, extra)) throw std::invalid_argument("unexpected trailing quantization token");
    if (!file.eof()) throw std::invalid_argument("quantization read failed");
}

template<class T> void array_json(const std::vector<T>& values) {
    std::cout << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) std::cout << ',';
        std::cout << static_cast<std::int64_t>(values[i]);
    }
    std::cout << ']';
}

void report(const npu::Config& c, const npu::Accelerator& model, double seconds, bool dump) {
    const auto& s = model.statistics();
    const auto& q = c.quant;
    std::cout << std::setprecision(17) << std::boolalpha;
    std::cout << "{\n\"schema_version\":1,\n\"config\":{"
        << "\"m\":" << c.m << ",\"n\":" << c.n << ",\"k\":" << c.k
        << ",\"tile_m\":" << c.tile_m << ",\"tile_n\":" << c.tile_n
        << ",\"tile_k\":" << c.tile_k << ",\"weight_bits\":" << c.weight_bits
        << ",\"array_rows\":" << c.array_rows << ",\"array_cols\":" << c.array_cols
        << ",\"macs_per_cycle\":" << c.macs_per_cycle
        << ",\"memory_bytes_per_cycle\":" << c.memory_bytes_per_cycle
        << ",\"memory_latency\":" << c.memory_latency
        << ",\"dma_burst_bytes\":" << c.dma_burst_bytes
        << ",\"weight_cache_bytes\":" << c.weight_cache_bytes
        << ",\"pipeline_cycles\":" << c.pipeline_cycles
        << ",\"buffer_slots\":" << c.buffer_slots << ",\"sram_bytes\":" << c.sram_bytes
        << ",\"seed\":" << c.seed << ",\"frequency_mhz\":" << c.frequency_mhz
        << ",\"timing\":\"" << c.timing << "\",\"quant\":{"
        << "\"input_zero\":" << q.input_zero << ",\"weight_zero\":" << q.weight_zero
        << ",\"output_zero\":" << q.output_zero << ",\"multiplier\":" << q.multiplier
        << ",\"shift\":" << q.shift << ",\"relu\":" << q.relu << ",\"rounding\":\"" << q.rounding << '"'
        << ",\"weight_zero_points\":";
    array_json(q.weight_zero_points);
    std::cout << ",\"multipliers\":"; array_json(q.multipliers);
    std::cout << ",\"shifts\":"; array_json(q.shifts);
    std::cout << "}},\n"
        << "\"cycles\":" << s.cycles << ",\n\"macs\":" << s.macs
        << ",\n\"read_bytes\":" << s.read_bytes << ",\n\"write_bytes\":" << s.write_bytes
        << ",\n\"compute_busy_cycles\":" << s.compute_busy_cycles
        << ",\n\"dma_busy_cycles\":" << s.dma_busy_cycles
        << ",\n\"peak_inflight_tiles\":" << s.peak_inflight_tiles
        << ",\n\"slot_wait_cycles\":" << s.slot_wait_cycles
        << ",\n\"reduction_blocks\":" << npu::reduction_blocks(c)
        << ",\n\"effective_macs_per_cycle\":" << npu::peak_macs_per_cycle(c)
        << ",\n\"dma_requests\":" << s.dma_requests << ",\n\"dma_bursts\":" << s.dma_bursts
        << ",\n\"weight_cache_hits\":" << s.weight_cache_hits
        << ",\n\"weight_cache_misses\":" << s.weight_cache_misses
        << ",\n\"weight_bytes_saved\":" << s.weight_bytes_saved
        << ",\n\"peak_cached_weight_bytes\":" << s.peak_cached_weight_bytes
        << ",\n\"scratchpad_bytes\":" << npu::scratchpad_bytes(c)
        << ",\n\"output_hash\":" << npu::output_hash(model.output())
        << ",\n\"numerical_match\":true"
        << ",\n\"simulation_wall_seconds\":" << seconds
        << ",\n\"compute_utilization\":" << static_cast<double>(s.compute_busy_cycles) / static_cast<double>(s.cycles)
        << ",\n\"memory_utilization\":" << static_cast<double>(s.dma_busy_cycles) / static_cast<double>(s.cycles)
        << ",\n\"mac_utilization\":" << static_cast<double>(s.macs) /
            (static_cast<double>(npu::peak_macs_per_cycle(c)) * static_cast<double>(s.cycles))
        << ",\n\"dma_payload_utilization\":" << static_cast<double>(s.read_bytes + s.write_bytes) /
            (static_cast<double>(c.memory_bytes_per_cycle) * static_cast<double>(s.cycles))
        << ",\n\"arithmetic_intensity_macs_per_byte\":" << static_cast<double>(s.macs) /
            static_cast<double>(s.read_bytes + s.write_bytes)
        << ",\n\"resource_lower_bound_cycles\":" << std::max(s.compute_busy_cycles, s.dma_busy_cycles)
        << ",\n\"modeled_latency_ns\":" << static_cast<double>(s.cycles) * 1000.0 / c.frequency_mhz;
    if (dump) {
        std::cout << ",\n\"accumulators\":"; array_json(model.output().accumulators);
        std::cout << ",\n\"outputs\":"; array_json(model.output().quantized);
    }
    std::cout << "\n}\n";
}
}  // namespace

int sc_main(int argc, char* argv[]) {
    try {
        auto args = parse(argc, argv);
        if (args.help) { usage(); return 0; }
        const auto problem = input_problem(args);
        input_quantization(args);
        npu::validate(args.config);
        const auto expected = npu::reference(problem, args.config.quant);
        sc_core::sc_set_time_resolution(1, sc_core::SC_NS);
        npu::Accelerator model("npu", args.config, problem);
        const auto before = std::chrono::steady_clock::now();
        sc_core::sc_start();
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - before).count();
        if (!model.completed()) throw std::runtime_error("model deadlocked before completion");
        if (model.output().accumulators != expected.accumulators || model.output().quantized != expected.quantized)
            throw std::runtime_error("numerical validation failed");
        if (!args.config.trace_path.empty()) {
            std::ofstream trace(args.config.trace_path);
            if (!trace) throw std::runtime_error("cannot open trace output");
            npu::write_trace(trace, model.trace());
        }
        report(args.config, model, seconds, args.dump);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "npu-sim: " << error.what() << '\n';
        return 2;
    }
}
