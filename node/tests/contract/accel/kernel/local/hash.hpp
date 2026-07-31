#pragma once

#include <kernel/program/compute/dsl.hpp>
#include <array>
#include <cstdint>

namespace node_accel_contract::kernel_case {

[[nodiscard]] inline std::uint64_t HashFixedLane32(
    const std::array<rund::kernel::i32, 8u>& values) noexcept {
  std::uint64_t hash = 1469598103934665603ull;
  for (const rund::kernel::i32 value : values) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value);
    for (std::uint32_t byte = 0u; byte < 4u; ++byte) {
      hash ^= (bits >> (byte * 8u)) & 0xffu;
      hash *= 1099511628211ull;
    }
  }
  return hash;
}

}  // namespace node_accel_contract::kernel_case
