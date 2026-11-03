#include "npu/core.hpp"

#include <systemc>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace sc_core;
using U64 = std::uint64_t;

void diagnostic_report(const sc_report& report, const sc_actions& actions) {
    if (actions & SC_DISPLAY) std::cerr << report.what() << '\n';
    sc_report_handler::default_handler(report, actions & ~SC_DISPLAY);
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

U64 integer(const std::string& text) {
    U64 value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        throw std::invalid_argument("expected unsigned integer: " + text);
    return value;
}

double coefficient(const std::string& text) {
    std::size_t end = 0;
    const auto value = std::stod(text, &end);
    if (end != text.size() || !std::isfinite(value) || value < 0 || value > 1.0e9)
        throw std::invalid_argument("energy coefficient must be finite and between 0 and 1e9");
    return value;
}

std::vector<std::string> split(const std::string& text, char separator) {
    std::vector<std::string> result;
    std::size_t begin = 0;
    while (true) {
        const auto end = text.find(separator, begin);
        const auto item = text.substr(begin, end == std::string::npos ? end : end - begin);
        if (item.empty()) throw std::invalid_argument("empty list item");
        result.push_back(item);
        if (end == std::string::npos) return result;
        begin = end + 1;
    }
}

struct Opp { U64 mhz = 0, mv = 0; };
struct Options {
    U64 cores = 2, jobs = 6, m = 4, n = 8, k = 16, models = 1, seed = 1;
    U64 arrival_ns = 0, batch_size = 6, deadline_ns = 0;
    U64 bus_bytes_per_ns = 16, bus_latency_ns = 8, macs_per_cycle = 16, pipeline_cycles = 4;
    U64 sram_bytes = 65536, cache_bytes = 4096;
    U64 gate_after_ns = 100, wakeup_ns = 12, switch_ns = 8, opp = 1, reference_mv = 800;
    std::vector<Opp> opps{{100, 700}, {200, 800}, {400, 900}};
    std::vector<U64> priorities{0, 2, 1};
    std::string governor = "fixed", vcd;
    bool dump = false, energy_provided = false;
    double mac_pj = 0, dram_byte_pj = 0, leakage_mw = 0, retention_mw = 0;
    double gated_mw = 0, wakeup_pj = 0, switch_pj = 0;
};

double nanoseconds(sc_time time) { return time.to_seconds() * 1.0e9; }
sc_time ns(double value) { return sc_time(value, SC_NS); }

Options parse_options(int argc, char* argv[]) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string flag(argv[i]);
        if (flag == "--dump-output") { options.dump = true; continue; }
        if (i + 1 == argc) throw std::invalid_argument("missing value for " + flag);
        const std::string value(argv[++i]);
        if (flag == "--cores") options.cores = integer(value);
        else if (flag == "--jobs") options.jobs = integer(value);
        else if (flag == "--m") options.m = integer(value);
        else if (flag == "--n") options.n = integer(value);
        else if (flag == "--k") options.k = integer(value);
        else if (flag == "--models") options.models = integer(value);
        else if (flag == "--seed") options.seed = integer(value);
        else if (flag == "--arrival-ns") options.arrival_ns = integer(value);
        else if (flag == "--batch-size") options.batch_size = integer(value);
        else if (flag == "--deadline-ns") options.deadline_ns = integer(value);
        else if (flag == "--bus-bytes-per-ns") options.bus_bytes_per_ns = integer(value);
        else if (flag == "--bus-latency-ns") options.bus_latency_ns = integer(value);
        else if (flag == "--macs-per-cycle") options.macs_per_cycle = integer(value);
        else if (flag == "--pipeline-cycles") options.pipeline_cycles = integer(value);
        else if (flag == "--sram-bytes") options.sram_bytes = integer(value);
        else if (flag == "--cache-bytes") options.cache_bytes = integer(value);
        else if (flag == "--gate-after-ns") options.gate_after_ns = integer(value);
        else if (flag == "--wakeup-ns") options.wakeup_ns = integer(value);
        else if (flag == "--switch-ns") options.switch_ns = integer(value);
        else if (flag == "--opp") options.opp = integer(value);
        else if (flag == "--reference-mv") options.reference_mv = integer(value);
        else if (flag == "--governor") options.governor = value;
        else if (flag == "--vcd") { if (value.empty()) throw std::invalid_argument("empty VCD path"); options.vcd = value; }
        else if (flag == "--priorities") {
            options.priorities.clear();
            for (const auto& item : split(value, ',')) options.priorities.push_back(integer(item));
        } else if (flag == "--opps") {
            options.opps.clear();
            for (const auto& item : split(value, ',')) {
                const auto pair = split(item, ':');
                if (pair.size() != 2) throw std::invalid_argument("OPP must be MHz:mV");
                options.opps.push_back({integer(pair[0]), integer(pair[1])});
            }
        } else {
            const double amount = coefficient(value);
            if (flag == "--mac-pj") options.mac_pj = amount;
            else if (flag == "--dram-byte-pj") options.dram_byte_pj = amount;
            else if (flag == "--leakage-mw") options.leakage_mw = amount;
            else if (flag == "--retention-mw") options.retention_mw = amount;
            else if (flag == "--gated-mw") options.gated_mw = amount;
            else if (flag == "--wakeup-pj") options.wakeup_pj = amount;
            else if (flag == "--switch-pj") options.switch_pj = amount;
            else throw std::invalid_argument("unknown argument: " + flag);
            options.energy_provided = true;
        }
    }
    require(options.cores >= 1 && options.cores <= 8, "cores must be between 1 and 8");
    require(options.jobs >= 1 && options.jobs <= 128, "jobs must be between 1 and 128");
    require(options.models >= 1 && options.models <= options.jobs, "models must be between 1 and jobs");
    require(options.m >= 1 && options.m <= 64 && options.n >= 1 && options.n <= 64 &&
            options.k >= 1 && options.k <= 256, "shape bounds are m,n 1..64 and k 1..256");
    require(options.batch_size >= 1 && options.batch_size <= 128, "batch-size must be between 1 and 128");
    require(options.seed <= std::numeric_limits<std::uint32_t>::max(), "seed must fit UINT32");
    require(options.arrival_ns <= 1000000 && options.deadline_ns <= 10000000, "arrival/deadline interval too large");
    require(options.bus_bytes_per_ns >= 1 && options.bus_bytes_per_ns <= 4096, "bus bandwidth must be 1..4096 bytes/ns");
    require(options.bus_latency_ns <= 1000000 && options.pipeline_cycles <= 1000000,
            "bus latency/pipeline bound exceeded");
    require(options.macs_per_cycle >= 1 && options.macs_per_cycle <= 1048576, "MAC throughput must be 1..1048576");
    require(options.gate_after_ns <= 10000000 && options.wakeup_ns <= 1000000 && options.switch_ns <= 1000000,
            "power transition bound exceeded");
    require(!options.opps.empty() && options.opps.size() <= 16 && options.opp < options.opps.size(), "invalid OPP selection");
    U64 previous = 0, previous_mv = 0;
    for (const auto& opp : options.opps) {
        require(opp.mhz >= 1 && opp.mhz <= 2000 && opp.mhz > previous, "OPP frequencies must increase within 1..2000 MHz");
        require(opp.mv >= 100 && opp.mv <= 2000, "OPP voltage must be 100..2000 mV");
        require(opp.mv >= previous_mv, "OPP voltages must not decrease with frequency");
        previous = opp.mhz;
        previous_mv = opp.mv;
    }
    require(options.reference_mv >= 100 && options.reference_mv <= 2000, "reference voltage must be 100..2000 mV");
    require(options.governor == "fixed" || options.governor == "deadline", "governor must be fixed or deadline");
    for (const auto priority : options.priorities) require(priority <= 255, "priority must be 0..255");
    require(options.sram_bytes <= 16777216 && options.cache_bytes <= options.sram_bytes, "SRAM/cache bound exceeded");
    const U64 weights = options.k * options.n + 4 * options.n;
    const U64 working = options.m * options.k + 5 * options.m * options.n;
    require(working + std::max(weights, options.cache_bytes) <= options.sram_bytes,
            "SRAM cannot hold activations, outputs, bias and weight residency");
    return options;
}

struct BusRecord {
    U64 core = 0, job = 0, bytes = 0;
    bool write = false;
    double requested_ns = 0, begin_ns = 0, end_ns = 0;
};

// Every client blocks on its own completion event. The single bus process
// serializes timed transfers, so two cores can never claim overlapping DRAM.
class SharedBus final : public sc_module {
public:
    sc_signal<bool> busy{"busy"};
    sc_signal<sc_dt::sc_uint<32>> owner{"owner"};
    std::vector<BusRecord> records;

    SC_HAS_PROCESS(SharedBus);
    SharedBus(sc_module_name name, const Options& options) : sc_module(name), options_(options) { SC_THREAD(run); }

    void transfer(U64 core, U64 job, bool write, U64 bytes) {
        Request request;
        request.record = {core, job, bytes, write, nanoseconds(sc_time_stamp()), 0, 0};
        queue_.push_back(&request);
        available_.notify(SC_ZERO_TIME);
        wait(request.done);
    }

    double duration(U64 bytes) const {
        return static_cast<double>(options_.bus_latency_ns + npu::ceil_div(bytes, options_.bus_bytes_per_ns));
    }

private:
    struct Request { BusRecord record; sc_event done; };
    const Options& options_;
    std::deque<Request*> queue_;
    sc_event available_;

    void run() {
        busy.write(false);
        owner.write(0);
        while (true) {
            if (queue_.empty()) { wait(available_); continue; }
            auto* request = queue_.front();
            queue_.pop_front();
            request->record.begin_ns = nanoseconds(sc_time_stamp());
            busy.write(true);
            owner.write(static_cast<unsigned>(request->record.core + 1));
            wait(ns(duration(request->record.bytes)));
            request->record.end_ns = nanoseconds(sc_time_stamp());
            records.push_back(request->record);
            busy.write(false);
            owner.write(0);
            request->done.notify(SC_ZERO_TIME);
        }
    }
};

struct Job {
    U64 id = 0, model = 0, priority = 0, release_ns = 0, deadline_ns = 0;
    U64 core = 0, opp = 0, compute_cycles = 0;
    double dispatch_ns = 0, compute_begin_ns = 0, compute_end_ns = 0, finish_ns = 0;
    bool cache_hit = false, woke = false, switched = false, numerical_match = false;
    npu::Problem source;
    npu::Output expected, output;
};

struct CacheEntry { U64 model = 0; std::vector<std::int8_t> weights; std::vector<std::int32_t> bias; };
struct CoreStats {
    double active_ns = 0, retention_ns = 0, gated_ns = 0;
    double last_finish_ns = 0, dynamic_pj = 0;
    U64 jobs = 0, wakeups = 0, transitions = 0, cache_hits = 0, cache_misses = 0;
    U64 cache_resident = 0, peak_cache = 0;
};

class Platform;
class Core final : public sc_module {
public:
    sc_signal<sc_dt::sc_uint<8>> state{"state"};
    sc_signal<sc_dt::sc_uint<32>> job_signal{"job"}, frequency{"frequency_mhz"}, voltage{"voltage_mv"};
    sc_event assigned;
    Job* job = nullptr;
    bool occupied = false;
    CoreStats stats;

    SC_HAS_PROCESS(Core);
    Core(sc_module_name name, Platform& platform, U64 id, const Options& options, SharedBus& bus);
    void account_final_idle(double finish);

private:
    Platform& platform_;
    U64 id_, current_opp_;
    const Options& options_;
    SharedBus& bus_;
    bool gated_ = true;
    std::deque<CacheEntry> cache_;
    void run();
    void account_idle(double until);
    U64 choose_opp(const Job& next, bool cache_hit) const;
};

class Platform final : public sc_module {
public:
    const Options& options;
    SharedBus bus;
    std::vector<Job> jobs;
    std::vector<std::unique_ptr<Core>> cores;
    std::vector<U64> dispatch_order;
    std::vector<Job*> ready;
    sc_event changed;
    sc_signal<sc_dt::sc_uint<32>> completed_signal{"completed_jobs"}, ready_signal{"ready_jobs"};
    U64 completed = 0;
    bool finished = false;

    SC_HAS_PROCESS(Platform);
    Platform(sc_module_name name, const Options& opts) : sc_module(name), options(opts), bus("dram", opts) {
        jobs.reserve(static_cast<std::size_t>(options.jobs));
        for (U64 id = 0; id < options.jobs; ++id) {
            Job next;
            next.id = id;
            next.model = id % options.models;
            next.priority = options.priorities[static_cast<std::size_t>(id % options.priorities.size())];
            next.release_ns = (id / options.batch_size) * options.arrival_ns;
            next.deadline_ns = options.deadline_ns ? next.release_ns + options.deadline_ns : 0;
            next.source.m = options.m; next.source.n = options.n; next.source.k = options.k;
            next.source.activations.resize(static_cast<std::size_t>(options.m * options.k));
            next.source.weights.resize(static_cast<std::size_t>(options.k * options.n));
            next.source.bias.resize(static_cast<std::size_t>(options.n));
            for (std::size_t i = 0; i < next.source.activations.size(); ++i)
                next.source.activations[i] = static_cast<std::int8_t>((i * 7 + options.seed + id * 3) % 31) - 15;
            for (std::size_t i = 0; i < next.source.weights.size(); ++i)
                next.source.weights[i] = static_cast<std::int8_t>((i * 11 + options.seed * 3 + next.model * 5) % 23) - 11;
            for (std::size_t i = 0; i < next.source.bias.size(); ++i)
                next.source.bias[i] = static_cast<std::int32_t>(i * 13 + next.model * 9) - 20;
            next.expected = npu::reference(next.source, quant());
            jobs.push_back(std::move(next));
        }
        for (U64 id = 0; id < options.cores; ++id)
            cores.push_back(std::make_unique<Core>(sc_gen_unique_name("core"), *this, id, options, bus));
        SC_THREAD(release);
        SC_THREAD(schedule);
    }

    static npu::Quantization quant() {
        npu::Quantization result;
        result.shift = 4;
        result.rounding = "even";
        return result;
    }

private:
    void release() {
        std::size_t index = 0;
        while (index < jobs.size()) {
            const auto release_time = ns(static_cast<double>(jobs[index].release_ns));
            if (release_time > sc_time_stamp()) wait(release_time - sc_time_stamp());
            const auto timestamp = jobs[index].release_ns;
            do { ready.push_back(&jobs[index++]); } while (index < jobs.size() && jobs[index].release_ns == timestamp);
            changed.notify(SC_ZERO_TIME);
        }
    }

    void schedule() {
        while (true) {
            // One delta gathers releases and completions at this timestamp
            // before priority selection; tie breaking is independent of core order.
            wait(changed);
            wait(SC_ZERO_TIME);
            if (completed == options.jobs) {
                completed_signal.write(static_cast<unsigned>(completed));
                ready_signal.write(0);
                finished = true;
                wait(SC_ZERO_TIME);
                wait(SC_ZERO_TIME);
                sc_stop();
                return;
            }
            std::stable_sort(ready.begin(), ready.end(), [](const Job* left, const Job* right) {
                if (left->priority != right->priority) return left->priority > right->priority;
                const auto ld = left->deadline_ns ? left->deadline_ns : std::numeric_limits<U64>::max();
                const auto rd = right->deadline_ns ? right->deadline_ns : std::numeric_limits<U64>::max();
                if (ld != rd) return ld < rd;
                return left->id < right->id;
            });
            for (auto& core : cores) {
                if (ready.empty()) break;
                if (core->occupied) continue;
                core->job = ready.front();
                ready.erase(ready.begin());
                core->occupied = true;
                dispatch_order.push_back(core->job->id);
                core->assigned.notify(SC_ZERO_TIME);
            }
            completed_signal.write(static_cast<unsigned>(completed));
            ready_signal.write(static_cast<unsigned>(ready.size()));
        }
    }
};

Core::Core(sc_module_name name, Platform& platform, U64 id, const Options& options, SharedBus& bus)
    : sc_module(name), platform_(platform), id_(id), current_opp_(options.opp), options_(options), bus_(bus) {
    SC_THREAD(run);
}

void Core::account_idle(double until) {
    const auto elapsed = std::max(0.0, until - stats.last_finish_ns);
    // Startup is powered down. Later idle retains state until the gate timer.
    const auto retained = stats.jobs == 0 ? 0.0 : std::min(elapsed, static_cast<double>(options_.gate_after_ns));
    stats.retention_ns += retained;
    stats.gated_ns += elapsed - retained;
}

void Core::account_final_idle(double finish) { account_idle(finish); }

U64 Core::choose_opp(const Job& next, bool cache_hit) const {
    if (options_.governor == "fixed" || next.deadline_ns == 0) return options_.opp;
    const auto macs = options_.m * options_.n * options_.k;
    const auto cycles = npu::ceil_div(macs, options_.macs_per_cycle) + options_.pipeline_cycles;
    const auto weights = cache_hit ? 0 : options_.k * options_.n + 4 * options_.n;
    const auto memory_ns = bus_.duration(options_.m * options_.k + weights) + bus_.duration(5 * options_.m * options_.n);
    const auto available = static_cast<double>(next.deadline_ns) - nanoseconds(sc_time_stamp());
    for (std::size_t index = 0; index < options_.opps.size(); ++index) {
        const auto& opp = options_.opps[index];
        const auto compute_ns = static_cast<double>(cycles) * 1000.0 / static_cast<double>(opp.mhz);
        const auto transition_ns = (gated_ ? options_.wakeup_ns : 0) + (index != current_opp_ ? options_.switch_ns : 0);
        if (memory_ns + compute_ns + static_cast<double>(transition_ns) <= available)
            return static_cast<U64>(index);
    }
    return static_cast<U64>(options_.opps.size() - 1);
}

void Core::run() {
    state.write(6);
    job_signal.write(0);
    frequency.write(static_cast<unsigned>(options_.opps[static_cast<std::size_t>(current_opp_)].mhz));
    voltage.write(static_cast<unsigned>(options_.opps[static_cast<std::size_t>(current_opp_)].mv));
    while (true) {
        if (!job) {
            if (gated_) wait(assigned);
            else {
                const auto gate_time = ns(stats.last_finish_ns + static_cast<double>(options_.gate_after_ns));
                if (gate_time > sc_time_stamp()) wait(gate_time - sc_time_stamp(), assigned);
                if (!job) { gated_ = true; state.write(6); continue; }
            }
        }
        if (!job) continue;
        Job& next = *job;
        next.core = id_;
        next.dispatch_ns = nanoseconds(sc_time_stamp());
        account_idle(next.dispatch_ns);
        job_signal.write(static_cast<unsigned>(next.id + 1));
        auto cached = std::find_if(cache_.begin(), cache_.end(), [&](const CacheEntry& entry) { return entry.model == next.model; });
        next.cache_hit = cached != cache_.end();
        next.opp = choose_opp(next, next.cache_hit);
        next.woke = gated_;
        if (gated_) {
            ++stats.wakeups;
            state.write(1);
            wait(ns(static_cast<double>(options_.wakeup_ns)));
            gated_ = false;
        }
        if (next.opp != current_opp_) {
            ++stats.transitions;
            next.switched = true;
            state.write(2);
            wait(ns(static_cast<double>(options_.switch_ns)));
            current_opp_ = next.opp;
        }
        const auto& opp = options_.opps[static_cast<std::size_t>(current_opp_)];
        frequency.write(static_cast<unsigned>(opp.mhz));
        voltage.write(static_cast<unsigned>(opp.mv));
        const U64 weight_bytes = options_.k * options_.n + 4 * options_.n;
        U64 read_bytes = options_.m * options_.k;
        if (next.cache_hit) ++stats.cache_hits;
        else { ++stats.cache_misses; read_bytes += weight_bytes; }
        state.write(3);
        bus_.transfer(id_, next.id, false, read_bytes);
        npu::Problem local;
        local.m = options_.m; local.n = options_.n; local.k = options_.k;
        local.activations = next.source.activations;
        if (next.cache_hit) {
            local.weights = cached->weights;
            local.bias = cached->bias;
            auto entry = std::move(*cached);
            cache_.erase(cached);
            cache_.push_back(std::move(entry));
        } else {
            local.weights = next.source.weights;
            local.bias = next.source.bias;
            if (weight_bytes <= options_.cache_bytes) {
                while (stats.cache_resident + weight_bytes > options_.cache_bytes) {
                    require(!cache_.empty(), "cache accounting underflow");
                    cache_.pop_front();
                    stats.cache_resident -= weight_bytes;
                }
                cache_.push_back({next.model, local.weights, local.bias});
                stats.cache_resident += weight_bytes;
                stats.peak_cache = std::max(stats.peak_cache, stats.cache_resident);
            }
        }
        next.output.accumulators.resize(static_cast<std::size_t>(options_.m * options_.n));
        next.output.quantized.resize(next.output.accumulators.size());
        const npu::Tile tile{0, 0, 0, options_.m, options_.n};
        next.compute_cycles = npu::ceil_div(options_.m * options_.n * options_.k, options_.macs_per_cycle) + options_.pipeline_cycles;
        next.compute_begin_ns = nanoseconds(sc_time_stamp());
        state.write(4);
        npu::compute_tile(local, Platform::quant(), tile, next.output);
        wait(ns(static_cast<double>(next.compute_cycles) * 1000.0 / static_cast<double>(opp.mhz)));
        next.compute_end_ns = nanoseconds(sc_time_stamp());
        state.write(5);
        bus_.transfer(id_, next.id, true, 5 * options_.m * options_.n);
        next.finish_ns = nanoseconds(sc_time_stamp());
        next.numerical_match = next.output.accumulators == next.expected.accumulators && next.output.quantized == next.expected.quantized;
        require(next.numerical_match, "core output differs from independent scalar reference");
        stats.active_ns += next.finish_ns - next.dispatch_ns;
        const auto voltage_ratio = static_cast<double>(opp.mv) / static_cast<double>(options_.reference_mv);
        stats.dynamic_pj += static_cast<double>(options_.m * options_.n * options_.k) * options_.mac_pj * voltage_ratio * voltage_ratio;
        ++stats.jobs;
        stats.last_finish_ns = next.finish_ns;
        state.write(0);
        job_signal.write(0);
        job = nullptr;
        occupied = false;
        ++platform_.completed;
        platform_.changed.notify(SC_ZERO_TIME);
    }
}

template<class T> void array(const std::vector<T>& values) {
    std::cout << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << static_cast<std::int64_t>(values[i]);
    }
    std::cout << ']';
}

void report(Platform& platform) {
    const auto& options = platform.options;
    const auto finish = nanoseconds(sc_time_stamp());
    U64 reads = 0, writes = 0, misses = 0, hits = 0, wakeups = 0, switches = 0, deadlines_missed = 0;
    double bus_ns = 0, queue_ns = 0, dynamic_pj = 0, active_ns = 0, retention_ns = 0, gated_ns = 0;
    for (const auto& transaction : platform.bus.records) {
        (transaction.write ? writes : reads) += transaction.bytes;
        bus_ns += transaction.end_ns - transaction.begin_ns;
        queue_ns += transaction.begin_ns - transaction.requested_ns;
    }
    for (auto& core : platform.cores) {
        core->account_final_idle(finish);
        const auto& stats = core->stats;
        active_ns += stats.active_ns;
        retention_ns += stats.retention_ns;
        gated_ns += stats.gated_ns;
        dynamic_pj += stats.dynamic_pj;
        hits += stats.cache_hits; misses += stats.cache_misses;
        wakeups += stats.wakeups; switches += stats.transitions;
    }
    for (const auto& job : platform.jobs)
        if (job.deadline_ns && job.finish_ns > static_cast<double>(job.deadline_ns)) ++deadlines_missed;
    const auto dram_pj = static_cast<double>(reads + writes) * options.dram_byte_pj;
    const auto active_pj = active_ns * options.leakage_mw;
    const auto retention_pj = retention_ns * options.retention_mw;
    const auto gated_pj = gated_ns * options.gated_mw;
    const auto transition_pj = static_cast<double>(wakeups) * options.wakeup_pj + static_cast<double>(switches) * options.switch_pj;
    std::cout << std::setprecision(15) << "{\"schema_version\":1,\"model\":\"systemc-mobile-multicore\","
              << "\"synthetic_architecture\":true,\"numerical_match\":true,\"preemptive\":false,"
              << "\"cores\":" << options.cores << ",\"jobs_completed\":" << platform.completed
              << ",\"m\":" << options.m << ",\"n\":" << options.n << ",\"k\":" << options.k
              << ",\"models\":" << options.models << ",\"seed\":" << options.seed
              << ",\"governor\":\"" << options.governor << "\",\"simulation_ns\":" << finish
              << ",\"bus_read_bytes\":" << reads << ",\"bus_write_bytes\":" << writes
              << ",\"bus_busy_ns\":" << bus_ns << ",\"bus_wait_ns\":" << queue_ns
              << ",\"cache_hits\":" << hits << ",\"cache_misses\":" << misses
              << ",\"wakeup_count\":" << wakeups << ",\"opp_transitions\":" << switches
              << ",\"deadline_misses\":" << deadlines_missed
              << ",\"configuration\":{\"arrival_ns\":" << options.arrival_ns << ",\"batch_size\":" << options.batch_size
              << ",\"relative_deadline_ns\":" << options.deadline_ns << ",\"bus_bytes_per_ns\":" << options.bus_bytes_per_ns
              << ",\"bus_latency_ns\":" << options.bus_latency_ns << ",\"macs_per_cycle\":" << options.macs_per_cycle
              << ",\"pipeline_cycles\":" << options.pipeline_cycles << ",\"sram_bytes_per_core\":" << options.sram_bytes
              << ",\"cache_bytes_per_core\":" << options.cache_bytes << ",\"gate_after_ns\":" << options.gate_after_ns
              << ",\"wakeup_ns\":" << options.wakeup_ns << ",\"opp_switch_ns\":" << options.switch_ns
              << ",\"initial_opp\":" << options.opp << ",\"compute_gate_retains_sram\":true}"
              << ",\"energy\":";
    if (options.energy_provided) {
        std::cout << "{\"coefficients_provided\":true,\"silicon_calibrated\":false,\"reference_mv\":" << options.reference_mv
                  << ",\"coefficients\":{\"mac_pj\":" << options.mac_pj << ",\"dram_byte_pj\":" << options.dram_byte_pj
                  << ",\"leakage_mw\":" << options.leakage_mw << ",\"retention_mw\":" << options.retention_mw
                  << ",\"gated_mw\":" << options.gated_mw << ",\"wakeup_pj\":" << options.wakeup_pj
                  << ",\"switch_pj\":" << options.switch_pj << '}'
                  << ",\"compute_pj\":" << dynamic_pj << ",\"dram_pj\":" << dram_pj
                  << ",\"active_leakage_pj\":" << active_pj << ",\"retention_pj\":" << retention_pj
                  << ",\"gated_pj\":" << gated_pj << ",\"transition_pj\":" << transition_pj
                  << ",\"total_pj\":" << dynamic_pj + dram_pj + active_pj + retention_pj + gated_pj + transition_pj << '}';
    } else std::cout << "null";
    std::cout << ",\"opps\":[";
    for (std::size_t i = 0; i < options.opps.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << "{\"mhz\":" << options.opps[i].mhz << ",\"mv\":" << options.opps[i].mv << '}';
    }
    std::cout << "],\"dispatch_order\":";
    array(platform.dispatch_order);
    std::cout << ",\"core_stats\":[";
    for (std::size_t i = 0; i < platform.cores.size(); ++i) {
        if (i) std::cout << ',';
        const auto& stats = platform.cores[i]->stats;
        std::cout << "{\"id\":" << i << ",\"jobs\":" << stats.jobs << ",\"active_ns\":" << stats.active_ns
                  << ",\"retention_ns\":" << stats.retention_ns << ",\"gated_ns\":" << stats.gated_ns
                  << ",\"peak_cache_bytes\":" << stats.peak_cache << '}';
    }
    std::cout << "],\"bus_transactions\":[";
    for (std::size_t i = 0; i < platform.bus.records.size(); ++i) {
        if (i) std::cout << ',';
        const auto& tx = platform.bus.records[i];
        std::cout << "{\"core\":" << tx.core << ",\"job\":" << tx.job << ",\"write\":" << (tx.write ? "true" : "false")
                  << ",\"bytes\":" << tx.bytes << ",\"requested_ns\":" << tx.requested_ns
                  << ",\"begin_ns\":" << tx.begin_ns << ",\"end_ns\":" << tx.end_ns << '}';
    }
    std::cout << "],\"jobs\":[";
    for (std::size_t i = 0; i < platform.jobs.size(); ++i) {
        if (i) std::cout << ',';
        const auto& job = platform.jobs[i];
        std::cout << "{\"id\":" << job.id << ",\"model\":" << job.model << ",\"core\":" << job.core
                  << ",\"priority\":" << job.priority << ",\"release_ns\":" << job.release_ns
                  << ",\"deadline_ns\":" << job.deadline_ns << ",\"dispatch_ns\":" << job.dispatch_ns
                  << ",\"compute_begin_ns\":" << job.compute_begin_ns << ",\"compute_end_ns\":" << job.compute_end_ns
                  << ",\"finish_ns\":" << job.finish_ns << ",\"response_ns\":" << job.finish_ns - static_cast<double>(job.release_ns)
                  << ",\"compute_cycles\":" << job.compute_cycles << ",\"opp\":" << job.opp
                  << ",\"cache_hit\":" << (job.cache_hit ? "true" : "false")
                  << ",\"woke\":" << (job.woke ? "true" : "false")
                  << ",\"numerical_match\":true,\"output_hash\":\"" << npu::output_hash(job.output) << '\"';
        if (options.dump) {
            std::cout << ",\"accumulators\":"; array(job.output.accumulators);
            std::cout << ",\"outputs\":"; array(job.output.quantized);
        }
        std::cout << '}';
    }
    std::cout << "]}\n";
}
}  // namespace

int sc_main(int argc, char* argv[]) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: npu-mobile-demo [options]\n"
                      << "  --cores 1..8 --jobs 1..128 --models N --m 1..64 --n 1..64 --k 1..256 --seed UINT32\n"
                      << "  --arrival-ns N --batch-size N --priorities 0,2,1 --deadline-ns N\n"
                      << "  --opps 100:700,200:800,400:900 --opp INDEX --governor fixed|deadline\n"
                      << "  --bus-bytes-per-ns N --bus-latency-ns N --macs-per-cycle N --pipeline-cycles N\n"
                      << "  --sram-bytes N --cache-bytes N --gate-after-ns N --wakeup-ns N --switch-ns N\n"
                      << "  --mac-pj N --dram-byte-pj N --leakage-mw N --retention-mw N --gated-mw N\n"
                      << "  --wakeup-pj N --switch-pj N --reference-mv N --dump-output --vcd BASENAME\n"
                      << "All defaults are synthetic architecture assumptions; energy defaults to zero/unconfigured.\n";
            return 0;
        }
        const auto options = parse_options(argc, argv);
        sc_report_handler::set_handler(diagnostic_report);
        sc_report_handler::set_actions("/OSCI/SystemC", SC_INFO, SC_DO_NOTHING);
        sc_report_handler::set_actions("tracing timescale unit set", SC_INFO, SC_DO_NOTHING);
        sc_report_handler::set_actions(SC_INFO, SC_DO_NOTHING);
        sc_set_time_resolution(1, SC_PS);
        Platform platform("mobile", options);
        std::unique_ptr<sc_trace_file, decltype(&sc_close_vcd_trace_file)>
            trace(nullptr, &sc_close_vcd_trace_file);
        if (!options.vcd.empty()) {
            trace.reset(sc_create_vcd_trace_file(options.vcd.c_str()));
            require(trace != nullptr, "cannot create VCD trace");
            trace->set_time_unit(1, SC_PS);
            sc_trace_delta_cycles(trace.get(), true);
            sc_trace(trace.get(), platform.bus.busy, "dram_busy");
            sc_trace(trace.get(), platform.bus.owner, "dram_owner");
            sc_trace(trace.get(), platform.completed_signal, "completed_jobs");
            sc_trace(trace.get(), platform.ready_signal, "ready_jobs");
            for (std::size_t i = 0; i < platform.cores.size(); ++i) {
                auto& core = *platform.cores[i];
                const auto prefix = "core" + std::to_string(i) + "_";
                sc_trace(trace.get(), core.state, prefix + "state");
                sc_trace(trace.get(), core.job_signal, prefix + "job");
                sc_trace(trace.get(), core.frequency, prefix + "frequency_mhz");
                sc_trace(trace.get(), core.voltage, prefix + "voltage_mv");
            }
        }
        // Worst-case bounded CLI configuration fits this simulated duration.
        // No real-time sleep is used: elapsed_ns comes only from SystemC waits.
        sc_start(sc_time(600, SC_SEC));
        trace.reset();
        require(platform.finished && platform.completed == options.jobs, "mobile platform timed out");
        report(platform);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "npu-mobile-demo: " << error.what() << '\n';
        return 2;
    }
}
