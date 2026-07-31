#pragma once

namespace rund::node::accel::cpu_simd_detail {
namespace {

[[nodiscard]] Mask Truthy(const Vec value) noexcept {
  return RUND_CPU_SIMD_NE(value, RUND_CPU_SIMD_SPLAT(0));
}

[[nodiscard]] Vec BooleanValue(const Mask mask) noexcept {
  return RUND_CPU_SIMD_SELECT(mask, RUND_CPU_SIMD_SPLAT(1),
                              RUND_CPU_SIMD_SPLAT(0));
}

[[nodiscard]] BitsVec Bits(const Vec value) noexcept {
  return std::bit_cast<BitsVec>(value);
}

[[nodiscard]] Vec SignedBits(const BitsVec value) noexcept {
  return std::bit_cast<Vec>(value);
}

}  // namespace
}  // namespace rund::node::accel::cpu_simd_detail
