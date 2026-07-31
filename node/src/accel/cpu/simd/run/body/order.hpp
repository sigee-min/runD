#pragma once

namespace rund::node::accel::cpu_simd_detail {
namespace {

[[nodiscard]] Mask ValueLt(const PreparedRun &, const Vec lhs,
                           const Vec rhs) noexcept {
  return RUND_CPU_SIMD_LT(lhs, rhs);
}

[[nodiscard]] Mask ValueLe(const PreparedRun &, const Vec lhs,
                           const Vec rhs) noexcept {
  return RUND_CPU_SIMD_LE(lhs, rhs);
}

[[nodiscard]] Mask ValueGt(const PreparedRun &, const Vec lhs,
                           const Vec rhs) noexcept {
  return RUND_CPU_SIMD_GT(lhs, rhs);
}

[[nodiscard]] Mask ValueGe(const PreparedRun &, const Vec lhs,
                           const Vec rhs) noexcept {
  return RUND_CPU_SIMD_GE(lhs, rhs);
}

[[nodiscard]] Vec ValueMin(const PreparedRun &prepared, const Vec lhs,
                           const Vec rhs) noexcept {
  return RUND_CPU_SIMD_VALUE_SELECT(ValueLt(prepared, lhs, rhs), lhs, rhs);
}

[[nodiscard]] Vec ValueMax(const PreparedRun &prepared, const Vec lhs,
                           const Vec rhs) noexcept {
  return RUND_CPU_SIMD_VALUE_SELECT(ValueGt(prepared, lhs, rhs), lhs, rhs);
}

} // namespace
} // namespace rund::node::accel::cpu_simd_detail
