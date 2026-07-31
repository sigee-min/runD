#pragma once

#include <accel/api.hpp>
#include <accel/device.hpp>

#include <node/accel/cpu/simd.hpp>

#include <kernel/program/compute/lowering/entry.hpp>
#include <kernel/program/compute/dsl.hpp>
#include <kernel/program/phase.hpp>

#include <math32/fixed/scalar.hpp>
#include <math64/fixed/scalar.hpp>

#include "test/assert.hpp"

#include <bit>
namespace node_accel_contract::cpu {

[[nodiscard]] constexpr rund::kernel::CpuCaps NeonCaps() noexcept {
  return rund::kernel::CpuCaps{
      .backend = rund::kernel::ComputeBackend::Cpu,
      .strategy = rund::kernel::CpuSimdStrategy::Neon,
      .lane_bytes = 16u,
      .fixed_lane32_lanes = 4u,
      .fixed_lane64_lanes = 2u,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] inline rund::AccelPolicy CpuOnlyPolicy() noexcept {
  rund::AccelPolicy policy{};
  policy.preferred[0] = rund::AccelApi::Cpu;
  policy.preferred_count = 1u;
  return policy;
}

[[nodiscard]] constexpr rund::kernel::i32
LogicalShiftRight32(const rund::kernel::i32 value,
                    const unsigned amount) noexcept {
  return std::bit_cast<rund::kernel::i32>(
      std::bit_cast<rund::kernel::u32>(value) >> amount);
}

[[nodiscard]] constexpr rund::kernel::i64
LogicalShiftRight64(const rund::kernel::i64 value,
                    const unsigned amount) noexcept {
  return std::bit_cast<rund::kernel::i64>(
      std::bit_cast<rund::kernel::u64>(value) >> amount);
}

[[nodiscard]] constexpr rund::kernel::i32
ArithmeticShiftRight32(const rund::kernel::i32 value,
                       const unsigned amount) noexcept {
  if (amount == 0u) {
    return value;
  }
  const rund::kernel::u32 bits = std::bit_cast<rund::kernel::u32>(value);
  const rund::kernel::u32 shifted = bits >> amount;
  const rund::kernel::u32 sign =
      (bits & 0x80000000u) == 0u ? 0u
                                 : (~rund::kernel::u32{0u} << (32u - amount));
  return std::bit_cast<rund::kernel::i32>(shifted | sign);
}

[[nodiscard]] constexpr rund::kernel::i64
ArithmeticShiftRight64(const rund::kernel::i64 value,
                       const unsigned amount) noexcept {
  if (amount == 0u) {
    return value;
  }
  const rund::kernel::u64 bits = std::bit_cast<rund::kernel::u64>(value);
  const rund::kernel::u64 shifted = bits >> amount;
  const rund::kernel::u64 sign =
      (bits & 0x8000000000000000ull) == 0ull
          ? 0ull
          : (~rund::kernel::u64{0u} << (64u - amount));
  return std::bit_cast<rund::kernel::i64>(shifted | sign);
}

} // namespace node_accel_contract::cpu
