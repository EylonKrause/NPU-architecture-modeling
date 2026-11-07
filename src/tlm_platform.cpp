#include "npu/core.hpp"
#include "npu/tlm_platform.hpp"

#include <systemc>
#include <tlm>
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/simple_target_socket.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace sc_core;
using namespace npu::tlm_platform;

std::uint32_t load_u32(const unsigned char* bytes) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i != 4; ++i)
        value |= static_cast<std::uint32_t>(bytes[i]) << (8U * i);
    return value;
}

void store_u32(unsigned char* bytes, std::uint32_t value) {
    for (unsigned i = 0; i != 4; ++i)
        bytes[i] = static_cast<unsigned char>((value >> (8U * i)) & 0xffU);
}

std::int32_t signed_word(std::uint32_t value) {
    if (value <= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
        return static_cast<std::int32_t>(value);
    return -1 - static_cast<std::int32_t>(std::numeric_limits<std::uint32_t>::max() - value);
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct Traffic {
    std::uint64_t transactions = 0, read_bytes = 0, write_bytes = 0;
};

// Targets never call wait(): initiators consume the annotated delay. Host and
// engine DMA share one reservation timeline, so their accesses serialise.
class Dram final : public sc_module {
public:
    tlm_utils::simple_target_socket<Dram> host_socket{"host_socket"};
    tlm_utils::simple_target_socket<Dram> dma_socket{"dma_socket"};
    Traffic host, dma;
    std::uint64_t protocol_errors = 0, debug_reads = 0, arbitration_wait_cycles = 0;

    Dram(sc_module_name name, sc_time period) : sc_module(name), period_(period), bytes_(dram_size, 0) {
        host_socket.register_b_transport(this, &Dram::host_transport);
        dma_socket.register_b_transport(this, &Dram::dma_transport);
        host_socket.register_transport_dbg(this, &Dram::debug_transport);
        dma_socket.register_transport_dbg(this, &Dram::debug_transport);
        host_socket.register_get_direct_mem_ptr(this, &Dram::no_dmi);
        dma_socket.register_get_direct_mem_ptr(this, &Dram::no_dmi);
    }

private:
    sc_time period_, available_ = SC_ZERO_TIME;
    std::vector<unsigned char> bytes_;

    bool valid(tlm::tlm_generic_payload& tx) {
        tx.set_dmi_allowed(false);
        if (tx.get_command() != tlm::TLM_READ_COMMAND && tx.get_command() != tlm::TLM_WRITE_COMMAND)
            tx.set_response_status(tlm::TLM_COMMAND_ERROR_RESPONSE);
        else if (tx.get_byte_enable_ptr()) tx.set_response_status(tlm::TLM_BYTE_ENABLE_ERROR_RESPONSE);
        else if (!tx.get_data_ptr() || tx.get_data_length() == 0 ||
                 tx.get_data_length() > maximum_burst || tx.get_streaming_width() < tx.get_data_length())
            tx.set_response_status(tlm::TLM_BURST_ERROR_RESPONSE);
        else if (tx.get_address() >= dram_size || tx.get_data_length() > dram_size - tx.get_address())
            tx.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE);
        else return true;
        ++protocol_errors;
        return false;
    }

    void transport(tlm::tlm_generic_payload& tx, sc_time& delay, Traffic& stats) {
        if (!valid(tx)) return;
        const auto count = tx.get_data_length();
        const auto arrival = sc_time_stamp() + delay;
        const auto begin = arrival > available_ ? arrival : available_;
        arbitration_wait_cycles += static_cast<std::uint64_t>((begin - arrival) / period_);
        const auto cycles = 3U + (count + 15U) / 16U;
        available_ = begin + period_ * static_cast<double>(cycles);
        delay = available_ - sc_time_stamp();
        auto* data = bytes_.data() + static_cast<std::size_t>(tx.get_address());
        if (tx.is_read()) {
            std::copy_n(data, count, tx.get_data_ptr());
            stats.read_bytes += count;
        } else {
            std::copy_n(tx.get_data_ptr(), count, data);
            stats.write_bytes += count;
        }
        ++stats.transactions;
        tx.set_response_status(tlm::TLM_OK_RESPONSE);
    }

    void host_transport(tlm::tlm_generic_payload& tx, sc_time& delay) { transport(tx, delay, host); }
    void dma_transport(tlm::tlm_generic_payload& tx, sc_time& delay) { transport(tx, delay, dma); }

    // Debug reads take no time, have no burst limit and stay out of the traffic
    // and protocol-error counts.
    unsigned debug_transport(tlm::tlm_generic_payload& tx) {
        if (tx.get_command() != tlm::TLM_READ_COMMAND) {
            tx.set_response_status(tlm::TLM_COMMAND_ERROR_RESPONSE);
            return 0;
        }
        if (!tx.get_data_ptr() || tx.get_address() >= dram_size ||
            tx.get_data_length() > dram_size - tx.get_address()) {
            tx.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE);
            return 0;
        }
        std::copy_n(bytes_.data() + static_cast<std::size_t>(tx.get_address()),
                    tx.get_data_length(), tx.get_data_ptr());
        ++debug_reads;
        tx.set_response_status(tlm::TLM_OK_RESPONSE);
        return tx.get_data_length();
    }

    bool no_dmi(tlm::tlm_generic_payload& tx, tlm::tlm_dmi&) {
        tx.set_dmi_allowed(false);
        return false;
    }
};

class Engine final : public sc_module {
public:
    sc_in<bool> clock{"clock"}, reset_n{"reset_n"};
    sc_out<bool> irq{"irq"};
    tlm_utils::simple_target_socket<Engine> registers{"registers"};
    tlm_utils::simple_initiator_socket<Engine> dma{"dma"};
    sc_signal<bool> busy_trace{"busy"}, done_trace{"done"};
    sc_signal<sc_dt::sc_uint<32>> jobs_trace{"jobs"};
    std::uint64_t protocol_errors = 0, register_transactions = 0, compute_cycles = 0;

    Engine(sc_module_name name, const npu::Config& config, sc_time period)
        : sc_module(name), config_(config), period_(period) {
        registers.register_b_transport(this, &Engine::mmio);
        registers.register_get_direct_mem_ptr(this, &Engine::no_dmi);
        SC_THREAD(run);
        sensitive << clock.pos();
        async_reset_signal_is(reset_n, false);
        SC_METHOD(observe);
        sensitive << changed_;
    }

private:
    const npu::Config& config_;
    sc_time period_;
    sc_event command_, changed_;
    // START is level-latched: a write in the reset-release timestep lands while
    // run() still waits for its first clock edge, so command_ alone would fire
    // before anyone waits on it.
    bool busy_ = false, done_ = false, irq_enabled_ = false, start_pending_ = false;
    sc_time start_time_ = SC_ZERO_TIME;
    std::uint32_t jobs_ = 0;

    void observe() {
        irq.write(done_ && irq_enabled_);
        busy_trace.write(busy_);
        done_trace.write(done_);
        jobs_trace.write(jobs_);
    }

    bool no_dmi(tlm::tlm_generic_payload& tx, tlm::tlm_dmi&) {
        tx.set_dmi_allowed(false);
        return false;
    }

    void reject(tlm::tlm_generic_payload& tx, tlm::tlm_response_status status) {
        ++protocol_errors;
        tx.set_response_status(status);
    }

    void mmio(tlm::tlm_generic_payload& tx, sc_time& delay) {
        tx.set_dmi_allowed(false);
        if (tx.get_command() != tlm::TLM_READ_COMMAND && tx.get_command() != tlm::TLM_WRITE_COMMAND)
            return reject(tx, tlm::TLM_COMMAND_ERROR_RESPONSE);
        if (tx.get_byte_enable_ptr()) return reject(tx, tlm::TLM_BYTE_ENABLE_ERROR_RESPONSE);
        if (!tx.get_data_ptr() || tx.get_data_length() != 4 || tx.get_streaming_width() < 4)
            return reject(tx, tlm::TLM_BURST_ERROR_RESPONSE);
        const auto address = tx.get_address();
        const auto* entry = (address & 3U) ? nullptr : find_register(address);
        if (!entry) return reject(tx, tlm::TLM_ADDRESS_ERROR_RESPONSE);
        if (!reset_n.read()) return reject(tx, tlm::TLM_GENERIC_ERROR_RESPONSE);
        if (entry->access == (tx.is_read() ? Access::write_only : Access::read_only))
            return reject(tx, tlm::TLM_COMMAND_ERROR_RESPONSE);
        if (tx.is_read()) {
            std::uint32_t value = 0;
            switch (address) {
            case reg_m: value = static_cast<std::uint32_t>(config_.shape.m); break;
            case reg_n: value = static_cast<std::uint32_t>(config_.shape.n); break;
            case reg_k: value = static_cast<std::uint32_t>(config_.shape.k); break;
            case reg_status: value = (busy_ ? status_busy : 0U) | (done_ ? status_done : 0U); break;
            case reg_irq_enable: value = irq_enabled_ ? 1U : 0U; break;
            case reg_jobs: value = jobs_; break;
            default: return reject(tx, tlm::TLM_COMMAND_ERROR_RESPONSE);
            }
            store_u32(tx.get_data_ptr(), value);
        } else {
            const auto value = load_u32(tx.get_data_ptr());
            switch (address) {
            case reg_start:
                if (value != 1U || busy_ || done_)
                    return reject(tx, tlm::TLM_GENERIC_ERROR_RESPONSE);
                busy_ = true;
                start_pending_ = true;
                start_time_ = sc_time_stamp() + delay + period_;
                command_.notify(delay + period_);
                break;
            case reg_irq_enable:
                if (value > 1U) return reject(tx, tlm::TLM_GENERIC_ERROR_RESPONSE);
                irq_enabled_ = value != 0;
                break;
            case reg_irq_ack:
                if (value != 1U || busy_) return reject(tx, tlm::TLM_GENERIC_ERROR_RESPONSE);
                done_ = false;
                break;
            default: return reject(tx, tlm::TLM_COMMAND_ERROR_RESPONSE);
            }
        }
        delay += period_;
        ++register_transactions;
        changed_.notify(SC_ZERO_TIME);
        tx.set_response_status(tlm::TLM_OK_RESPONSE);
    }

    void transfer(bool write, std::uint64_t address, unsigned char* bytes, std::size_t length) {
        for (std::size_t offset = 0; offset < length; offset += maximum_burst) {
            const auto count = static_cast<unsigned>(std::min<std::size_t>(maximum_burst, length - offset));
            tlm::tlm_generic_payload tx;
            tx.set_command(write ? tlm::TLM_WRITE_COMMAND : tlm::TLM_READ_COMMAND);
            tx.set_address(address + offset);
            tx.set_data_ptr(bytes + offset);
            tx.set_data_length(count);
            tx.set_streaming_width(count);
            tx.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
            sc_time delay = SC_ZERO_TIME;
            dma->b_transport(tx, delay);
            require(tx.is_response_ok(), "engine DMA failed");
            wait(delay);
        }
    }

    void run() {
        busy_ = false; done_ = false; irq_enabled_ = false; start_pending_ = false; jobs_ = 0;
        command_.cancel();
        changed_.notify(SC_ZERO_TIME);
        wait();
        while (true) {
            while (!start_pending_) wait(command_);
            start_pending_ = false;
            // A START latched before the first clock edge after reset is seen at
            // that edge; keep the issue time the MMIO write annotated.
            if (start_time_ > sc_time_stamp()) wait(start_time_ - sc_time_stamp());
            npu::Problem problem;
            problem.m = config_.shape.m; problem.n = config_.shape.n; problem.k = config_.shape.k;
            problem.activations.resize(static_cast<std::size_t>(config_.shape.m * config_.shape.k));
            problem.weights.resize(static_cast<std::size_t>(config_.shape.k * config_.shape.n));
            problem.bias.resize(static_cast<std::size_t>(config_.shape.n));
            transfer(false, activation_address,
                     reinterpret_cast<unsigned char*>(problem.activations.data()),
                     problem.activations.size());
            transfer(false, weight_address, reinterpret_cast<unsigned char*>(problem.weights.data()),
                     problem.weights.size());
            std::vector<unsigned char> bias(problem.bias.size() * 4U);
            transfer(false, bias_address, bias.data(), bias.size());
            for (std::size_t i = 0; i < problem.bias.size(); ++i)
                problem.bias[i] = signed_word(load_u32(bias.data() + 4U * i));
            npu::Output output;
            output.accumulators.resize(static_cast<std::size_t>(config_.shape.m * config_.shape.n));
            output.quantized.resize(output.accumulators.size());
            for (const auto& tile : npu::make_tiles(config_)) {
                npu::compute_tile(problem, config_.quant, tile, output);
                const auto cycles = npu::tile_compute_cycles(config_, tile);
                compute_cycles += cycles;
                wait(period_ * static_cast<double>(cycles));
            }
            transfer(true, output_address, reinterpret_cast<unsigned char*>(output.quantized.data()),
                     output.quantized.size());
            std::vector<unsigned char> accum(output.accumulators.size() * 4U);
            for (std::size_t i = 0; i < output.accumulators.size(); ++i)
                store_u32(accum.data() + 4U * i, static_cast<std::uint32_t>(output.accumulators[i]));
            transfer(true, accumulator_address, accum.data(), accum.size());
            busy_ = false; done_ = true; ++jobs_;
            changed_.notify(SC_ZERO_TIME);
        }
    }
};

struct JobResult {
    std::uint64_t id = 0, elapsed_cycles = 0, hash = 0;
    npu::Output output;
};

class Host final : public sc_module {
public:
    tlm_utils::simple_initiator_socket<Host> mmio{"mmio"}, memory{"memory"};
    sc_in<bool> clock{"clock"}, irq{"irq"};
    sc_out<bool> reset_n{"reset_n"};
    bool completed = false;
    std::uint64_t expected_errors = 0, irqs = 0;
    sc_time reset_release = SC_ZERO_TIME, first_start = SC_ZERO_TIME;
    std::vector<JobResult> results;

    Host(sc_module_name name, const npu::Config& config, unsigned jobs, sc_time period,
         bool start_at_reset_release)
        : sc_module(name), config_(config), jobs_(jobs), period_(period),
          start_at_reset_release_(start_at_reset_release) {
        SC_THREAD(run);
    }

private:
    const npu::Config& config_;
    unsigned jobs_;
    sc_time period_;
    bool start_at_reset_release_;

    void transaction(bool registers, tlm::tlm_command command, std::uint64_t address,
                     unsigned char* bytes, unsigned length,
                     tlm::tlm_response_status expected = tlm::TLM_OK_RESPONSE,
                     bool byte_enable = false, unsigned streaming_width = 0) {
        tlm::tlm_generic_payload tx;
        unsigned char enable = 0xff;
        tx.set_command(command); tx.set_address(address); tx.set_data_ptr(bytes);
        tx.set_data_length(length); tx.set_streaming_width(streaming_width ? streaming_width : length);
        if (byte_enable) { tx.set_byte_enable_ptr(&enable); tx.set_byte_enable_length(1); }
        tx.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
        sc_time delay = SC_ZERO_TIME;
        if (registers) mmio->b_transport(tx, delay); else memory->b_transport(tx, delay);
        require(tx.get_response_status() == expected, "unexpected TLM response status");
        require(!tx.is_dmi_allowed(), "target granted DMI");
        if (expected != tlm::TLM_OK_RESPONSE) ++expected_errors;
        if (delay != SC_ZERO_TIME) wait(delay);
    }

    void write_register(std::uint64_t address, std::uint32_t value,
                        tlm::tlm_response_status expected = tlm::TLM_OK_RESPONSE) {
        std::array<unsigned char, 4> bytes{}; store_u32(bytes.data(), value);
        transaction(true, tlm::TLM_WRITE_COMMAND, address, bytes.data(), 4, expected);
    }

    std::uint32_t read_register(std::uint64_t address) {
        std::array<unsigned char, 4> bytes{};
        transaction(true, tlm::TLM_READ_COMMAND, address, bytes.data(), 4);
        return load_u32(bytes.data());
    }

    void transfer(bool write, std::uint64_t address, unsigned char* bytes, std::size_t length) {
        for (std::size_t offset = 0; offset < length; offset += maximum_burst) {
            const auto count = static_cast<unsigned>(std::min<std::size_t>(maximum_burst, length - offset));
            transaction(false, write ? tlm::TLM_WRITE_COMMAND : tlm::TLM_READ_COMMAND,
                        address + offset, bytes + offset, count);
        }
    }

    // Each access must come back with the listed error response; DMI is refused.
    void protocol_checks() {
        std::array<unsigned char, 68> bytes{};
        transaction(true, tlm::TLM_READ_COMMAND, reg_status + 1, bytes.data(), 4,
                    tlm::TLM_ADDRESS_ERROR_RESPONSE);
        transaction(true, tlm::TLM_READ_COMMAND, 0x80, bytes.data(), 4, tlm::TLM_ADDRESS_ERROR_RESPONSE);
        transaction(true, tlm::TLM_WRITE_COMMAND, reg_m, bytes.data(), 4, tlm::TLM_COMMAND_ERROR_RESPONSE);
        transaction(true, tlm::TLM_READ_COMMAND, reg_status, bytes.data(), 4,
                    tlm::TLM_BYTE_ENABLE_ERROR_RESPONSE, true);
        transaction(true, tlm::TLM_IGNORE_COMMAND, reg_status, bytes.data(), 4,
                    tlm::TLM_COMMAND_ERROR_RESPONSE);
        transaction(true, tlm::TLM_READ_COMMAND, reg_status, bytes.data(), 2, tlm::TLM_BURST_ERROR_RESPONSE);
        transaction(false, tlm::TLM_READ_COMMAND, dram_size - 2, bytes.data(), 4,
                    tlm::TLM_ADDRESS_ERROR_RESPONSE);
        transaction(false, tlm::TLM_IGNORE_COMMAND, 0, bytes.data(), 4, tlm::TLM_COMMAND_ERROR_RESPONSE);
        transaction(false, tlm::TLM_WRITE_COMMAND, 0, bytes.data(), 4,
                    tlm::TLM_BYTE_ENABLE_ERROR_RESPONSE, true);
        transaction(false, tlm::TLM_READ_COMMAND, 0, bytes.data(), 65, tlm::TLM_BURST_ERROR_RESPONSE);
        transaction(false, tlm::TLM_READ_COMMAND, 0, bytes.data(), 4, tlm::TLM_BURST_ERROR_RESPONSE,
                    false, 2);
        tlm::tlm_generic_payload tx;
        tlm::tlm_dmi dmi;
        require(!memory->get_direct_mem_ptr(tx, dmi), "DRAM must reject DMI");
        require(!mmio->get_direct_mem_ptr(tx, dmi), "MMIO must reject DMI");
    }

    npu::Problem input(unsigned job) const {
        npu::Problem p;
        p.m = config_.shape.m; p.n = config_.shape.n; p.k = config_.shape.k;
        for (std::uint64_t i = 0; i < p.m * p.k; ++i)
            p.activations.push_back(static_cast<std::int8_t>(
                static_cast<int>((i * 7 + config_.shape.seed + job * 3U) % 31U) - 15));
        for (std::uint64_t i = 0; i < p.k * p.n; ++i)
            p.weights.push_back(static_cast<std::int8_t>(
                static_cast<int>((i * 11 + config_.shape.seed * 3 + job * 5U) % 23U) - 11));
        for (std::uint64_t i = 0; i < p.n; ++i)
            p.bias.push_back(static_cast<std::int32_t>(i * 13U + job * 9U) - 20);
        return p;
    }

    void stage(npu::Problem& p) {
        transfer(true, activation_address, reinterpret_cast<unsigned char*>(p.activations.data()),
                 p.activations.size());
        transfer(true, weight_address, reinterpret_cast<unsigned char*>(p.weights.data()), p.weights.size());
        std::vector<unsigned char> bias(p.bias.size() * 4U);
        for (std::size_t i = 0; i < p.bias.size(); ++i)
            store_u32(bias.data() + 4U * i, static_cast<std::uint32_t>(p.bias[i]));
        transfer(true, bias_address, bias.data(), bias.size());
    }

    // A debug read of the staged bias returns DRAM contents without advancing time.
    void check_debug_read(const npu::Problem& p) {
        std::array<unsigned char, 4> debug{};
        tlm::tlm_generic_payload debug_tx;
        debug_tx.set_command(tlm::TLM_READ_COMMAND); debug_tx.set_address(bias_address);
        debug_tx.set_data_ptr(debug.data()); debug_tx.set_data_length(4); debug_tx.set_streaming_width(4);
        const auto debug_before = sc_time_stamp();
        require(memory->transport_dbg(debug_tx) == 4 && debug_tx.is_response_ok() &&
                sc_time_stamp() == debug_before &&
                load_u32(debug.data()) == static_cast<std::uint32_t>(p.bias[0]),
                "debug read failed or consumed time");
    }

    void check_identity() {
        require(read_register(reg_m) == config_.shape.m && read_register(reg_n) == config_.shape.n &&
                read_register(reg_k) == config_.shape.k, "fixed dimensions readback failed");
    }

    // Job 0 completes with IRQ_ENABLE set. Later jobs complete masked, are
    // polled, and only then unmask the pending interrupt.
    void run_job(unsigned job, npu::Problem p) {
        const auto expected = npu::reference(p, config_.quant);
        const bool first = results.empty();
        const bool staged_in_reset = first && start_at_reset_release_;
        if (!staged_in_reset) stage(p);
        check_debug_read(p);
        const auto irq_enable = job == 0 ? 1U : 0U;
        if (!staged_in_reset) write_register(reg_irq_enable, irq_enable);
        const auto before = sc_time_stamp();
        if (first) first_start = before;
        write_register(reg_start, 1);
        if (staged_in_reset) write_register(reg_irq_enable, irq_enable);
        require(read_register(reg_status) == status_busy, "START did not set BUSY");
        write_register(reg_start, 1, tlm::TLM_GENERIC_ERROR_RESPONSE);
        // This scratch read contends with the engine's first DMA burst on the
        // shared DRAM timeline.
        std::array<unsigned char, 4> scratch{};
        transfer(false, 0x5000, scratch.data(), scratch.size());
        require(load_u32(scratch.data()) == 0, "unwritten DRAM scratch region changed");
        if (job == 0) {
            if (!irq.read()) wait(irq.posedge_event());
        } else {
            while ((read_register(reg_status) & status_done) == 0U) wait(clock.posedge_event());
            require(!irq.read(), "masked IRQ asserted");
            write_register(reg_irq_enable, 1);
            if (!irq.read()) wait(irq.posedge_event());
        }
        ++irqs;
        require(read_register(reg_status) == status_done, "IRQ asserted without completed output");
        require(read_register(reg_jobs) == job + 1U, "job completion count mismatch");
        JobResult result;
        result.id = job;
        result.elapsed_cycles = static_cast<std::uint64_t>((sc_time_stamp() - before) / period_);
        result.output.accumulators.resize(expected.accumulators.size());
        result.output.quantized.resize(expected.quantized.size());
        transfer(false, output_address, reinterpret_cast<unsigned char*>(result.output.quantized.data()),
                 result.output.quantized.size());
        std::vector<unsigned char> accum(expected.accumulators.size() * 4U);
        transfer(false, accumulator_address, accum.data(), accum.size());
        for (std::size_t i = 0; i < result.output.accumulators.size(); ++i)
            result.output.accumulators[i] = signed_word(load_u32(accum.data() + 4U * i));
        require(result.output.accumulators == expected.accumulators &&
                result.output.quantized == expected.quantized,
                "DMA result differs from the scalar reference");
        result.hash = npu::output_hash(result.output);
        results.push_back(std::move(result));
        write_register(reg_irq_ack, 1);
        require(!irq.read() && read_register(reg_status) == 0, "IRQ acknowledgement did not restore IDLE");
    }

    void run() {
        reset_n.write(false);
        auto boot = input(0);
        // DRAM sits outside the engine's reset domain, so a boot sequence can
        // stage the first job while the engine is held in reset.
        if (start_at_reset_release_) stage(boot);
        for (unsigned i = 0; i < 3; ++i) wait(clock.posedge_event());
        reset_n.write(true);
        reset_release = sc_time_stamp();
        if (start_at_reset_release_) {
            // The register decoder sees reset_n high one delta after the write;
            // START then lands in the reset-release timestep.
            wait(SC_ZERO_TIME);
            run_job(0, std::move(boot));
            check_identity();
            protocol_checks();
        } else {
            wait(clock.posedge_event());
            require(read_register(reg_status) == 0 && read_register(reg_jobs) == 0 && !irq.read(),
                    "reset state is incorrect");
            check_identity();
            protocol_checks();
            run_job(0, std::move(boot));
        }
        for (unsigned job = 1; job < jobs_; ++job) run_job(job, input(job));
        completed = true;
        sc_stop();
    }
};

template<class T> void print_array(const std::vector<T>& values) {
    std::cout << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << static_cast<std::int64_t>(values[i]);
    }
    std::cout << ']';
}


unsigned parse_unsigned(const std::string& flag, const std::string& text) {
    unsigned value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
        throw std::invalid_argument(flag + ": expected an unsigned 32-bit integer, got '" + text + "'");
    return value;
}

constexpr const char* usage =
    "Usage: npu-tlm-demo [--jobs 1..100] [--seed UINT32] [--dump-output] [--vcd BASENAME]\n"
    "                    [--start-at-reset-release]\n"
    "  --start-at-reset-release  stage job 0 while the engine is held in reset and write\n"
    "                            START in the reset-release timestep\n";
}  // namespace

int sc_main(int argc, char* argv[]) {
    try {
        for (int i = 1; i < argc; ++i)
            if (std::string(argv[i]) == "--help") { std::cout << usage; return 0; }
        unsigned jobs = 2, seed = 1;
        bool dump = false, start_at_reset_release = false;
        std::string vcd;
        for (int i = 1; i < argc; ++i) {
            const std::string flag(argv[i]);
            if (flag == "--dump-output") { dump = true; continue; }
            if (flag == "--start-at-reset-release") { start_at_reset_release = true; continue; }
            if (flag != "--jobs" && flag != "--seed" && flag != "--vcd")
                throw std::invalid_argument("unknown argument: " + flag);
            if (i + 1 == argc) throw std::invalid_argument(flag + ": missing value");
            const std::string value(argv[++i]);
            if (flag == "--jobs") jobs = parse_unsigned(flag, value);
            else if (flag == "--seed") seed = parse_unsigned(flag, value);
            else if (value.empty()) throw std::invalid_argument("--vcd: empty path");
            else vcd = value;
        }
        if (jobs == 0 || jobs > 100) throw std::invalid_argument("--jobs: must be between 1 and 100");
        npu::Config config;
        config.shape.m = gemm_m; config.shape.n = gemm_n; config.shape.k = gemm_k;
        config.mapping.tile_m = 2; config.mapping.tile_n = 3;
        config.arch.compute.macs_per_cycle = 4; config.arch.compute.pipeline_cycles = 2;
        config.quant.shift = 4; config.quant.rounding = npu::Rounding::HalfToEven;
        config.shape.seed = seed;
        npu::validate(config);
        // Keep stdout machine-readable. Warnings and errors keep their default
        // actions; only the routine stop and timescale notices are hidden.
        sc_report_handler::set_actions("/OSCI/SystemC", SC_INFO, SC_DO_NOTHING);
        sc_report_handler::set_actions("tracing timescale unit set", SC_INFO, SC_DO_NOTHING);
        sc_set_time_resolution(1, SC_PS);
        const sc_time period(5, SC_NS);
        sc_clock clock("clock", period);
        sc_signal<bool> reset_n("reset_n"), irq("irq");
        Dram dram("dram", period);
        Engine engine("npu", config, period);
        Host host("host", config, jobs, period, start_at_reset_release);
        engine.clock(clock); engine.reset_n(reset_n); engine.irq(irq);
        host.clock(clock); host.reset_n(reset_n); host.irq(irq);
        host.mmio.bind(engine.registers); host.memory.bind(dram.host_socket);
        engine.dma.bind(dram.dma_socket);
        sc_trace_file* trace = nullptr;
        if (!vcd.empty()) {
            trace = sc_create_vcd_trace_file(vcd.c_str());
            require(trace != nullptr, "cannot create VCD trace");
            trace->set_time_unit(1, SC_PS);
            sc_trace(trace, clock, "clock"); sc_trace(trace, reset_n, "reset_n"); sc_trace(trace, irq, "irq");
            sc_trace(trace, engine.busy_trace, "npu_busy"); sc_trace(trace, engine.done_trace, "npu_done");
            sc_trace(trace, engine.jobs_trace, "completed_jobs");
        }
        // Watchdog: a lost IRQ or START handshake ends the run here instead of hanging.
        sc_start(sc_time(1, SC_MS));
        if (trace) sc_close_vcd_trace_file(trace);
        require(host.completed && host.results.size() == jobs,
                "TLM platform timed out before all jobs completed");
        // The host checked each access's response status; the targets' error
        // counters must agree with its count of rejected accesses.
        require(engine.protocol_errors + dram.protocol_errors == host.expected_errors,
                "target error counters disagree with the host's rejected accesses");
        const auto cycle = [&](sc_time time) { return static_cast<std::uint64_t>(time / period); };
        std::cout << "{\n\"schema_version\":1,\"model\":\"systemc-tlm2-serial-coprocessor\","
                  << "\"jobs_completed\":" << host.results.size() << ",\"interrupts_observed\":" << host.irqs
                  << ",\"seed\":" << seed << ",\"m\":" << config.shape.m << ",\"n\":" << config.shape.n
                  << ",\"k\":" << config.shape.k << ",\"clock_period_ns\":" << period / sc_time(1, SC_NS)
                  << ",\"reset_release_cycle\":" << cycle(host.reset_release)
                  << ",\"first_start_cycle\":" << cycle(host.first_start)
                  << ",\"simulation_cycles\":" << cycle(sc_time_stamp())
                  << ",\"compute_cycles\":" << engine.compute_cycles
                  << ",\"mmio_transactions\":" << engine.register_transactions
                  << ",\"host_memory_transactions\":" << dram.host.transactions
                  << ",\"memory_arbitration_wait_cycles\":" << dram.arbitration_wait_cycles
                  << ",\"dma_transactions\":" << dram.dma.transactions
                  << ",\"dma_read_bytes\":" << dram.dma.read_bytes
                  << ",\"dma_write_bytes\":" << dram.dma.write_bytes
                  << ",\"debug_reads\":" << dram.debug_reads
                  << ",\"expected_protocol_errors\":" << host.expected_errors << ",\"jobs\":[";
        for (std::size_t i = 0; i < host.results.size(); ++i) {
            if (i) std::cout << ',';
            const auto& result = host.results[i];
            std::cout << "{\"id\":" << result.id << ",\"elapsed_cycles\":" << result.elapsed_cycles
                      << ",\"output_hash\":" << result.hash;
            if (dump) {
                std::cout << ",\"accumulators\":"; print_array(result.output.accumulators);
                std::cout << ",\"outputs\":"; print_array(result.output.quantized);
            }
            std::cout << '}';
        }
        std::cout << "]\n}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "npu-tlm-demo: " << error.what() << '\n';
        return 2;
    }
}
