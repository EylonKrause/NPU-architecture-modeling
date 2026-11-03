#pragma once

#include <cstdint>

namespace npu::tlm_platform {

// The demo models one fixed-function INT8 dense engine, not a commercial ISA.
// All registers are aligned, little-endian, 32-bit accesses.
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
constexpr std::uint64_t activation_address = 0x0000;
constexpr std::uint64_t weight_address = 0x1000;
constexpr std::uint64_t bias_address = 0x2000;
constexpr std::uint64_t output_address = 0x3000;
constexpr std::uint64_t accumulator_address = 0x4000;
constexpr std::uint64_t dram_size = 65536;
constexpr unsigned maximum_burst = 64;

}  // namespace npu::tlm_platform
