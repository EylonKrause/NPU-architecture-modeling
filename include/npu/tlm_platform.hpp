#pragma once

#include <array>
#include <cstdint>

namespace npu::tlm_platform {

// Fixed geometry of the INT8 GEMM engine: C[M][N] = A[M][K] * W[K][N] + bias[N].
constexpr std::uint32_t gemm_m = 3;
constexpr std::uint32_t gemm_n = 5;
constexpr std::uint32_t gemm_k = 7;

constexpr std::uint64_t reg_m = 0x00;
constexpr std::uint64_t reg_n = 0x04;
constexpr std::uint64_t reg_k = 0x08;
constexpr std::uint64_t reg_start = 0x0c;
constexpr std::uint64_t reg_status = 0x10;
constexpr std::uint64_t reg_irq_enable = 0x14;
constexpr std::uint64_t reg_irq_ack = 0x18;
constexpr std::uint64_t reg_jobs = 0x1c;
constexpr std::uint32_t status_busy = 1;
constexpr std::uint32_t status_done = 2;

enum class Access : std::uint8_t { read_only, write_only, read_write };

struct Register {
    std::uint64_t offset;
    const char* name;
    Access access;
    std::uint32_t reset;
    const char* description;
};

// MMIO register map. 32-bit aligned little-endian accesses only; any other
// width, a byte enable or an unmapped offset is an error response. A read of a
// write-only register or a write of a read-only one returns
// TLM_COMMAND_ERROR_RESPONSE. Every access is refused while reset_n is low.
inline constexpr std::array<Register, 8> register_map{{
    {reg_m, "M", Access::read_only, gemm_m, "output rows"},
    {reg_n, "N", Access::read_only, gemm_n, "output columns"},
    {reg_k, "K", Access::read_only, gemm_k, "reduction depth"},
    {reg_start, "START", Access::write_only, 0,
     "write 1: latch a job; refused while STATUS.BUSY or STATUS.DONE is set"},
    {reg_status, "STATUS", Access::read_only, 0,
     "bit 0 BUSY: job latched or running; bit 1 DONE: outputs in DRAM, cleared by IRQ_ACK"},
    {reg_irq_enable, "IRQ_ENABLE", Access::read_write, 0, "bit 0: irq = STATUS.DONE & IRQ_ENABLE"},
    {reg_irq_ack, "IRQ_ACK", Access::write_only, 0, "write 1: clear DONE; refused while BUSY"},
    {reg_jobs, "JOBS", Access::read_only, 0, "jobs completed since reset release"},
}};

constexpr const Register* find_register(std::uint64_t offset) {
    for (const auto& entry : register_map)
        if (entry.offset == offset) return &entry;
    return nullptr;
}

// DRAM map seen by both the host and the engine DMA. Reset does not clear DRAM.
constexpr std::uint64_t activation_address = 0x0000;   // int8 A[M][K], row-major
constexpr std::uint64_t weight_address = 0x1000;       // int8 W[K][N], row-major
constexpr std::uint64_t bias_address = 0x2000;         // int32 bias[N]
constexpr std::uint64_t output_address = 0x3000;       // int8 C[M][N]
constexpr std::uint64_t accumulator_address = 0x4000;  // int32 accumulators[M][N]
constexpr std::uint64_t dram_size = 65536;
constexpr unsigned maximum_burst = 64;

}  // namespace npu::tlm_platform
