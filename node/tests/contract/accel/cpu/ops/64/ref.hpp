#pragma once

#include "../../fixed.hpp"
#include "work.hpp"

namespace node_accel_contract::cpu::ops64 {

template <class... Values>
[[nodiscard]] rund::kernel::i64 SaturatedSum(const Values... values) noexcept {
  return rund::math64::detail::ScalarClamp(
      (rund::math64::detail::i128{0} + ... +
       static_cast<rund::math64::detail::i128>(values)));
}

[[nodiscard]] rund::kernel::i64
Saturate(const rund::kernel::i64 value) noexcept {
  constexpr rund::kernel::i64 fixed_max = 0x7fffffffffffffffll;
  return rund::math64::detail::ScalarClamp(value, 0, fixed_max);
}

[[nodiscard]] rund::kernel::i64 Step(const rund::kernel::i64 edge,
                                     const rund::kernel::i64 value) noexcept {
  return value < edge ? 0 : 0x7fffffffffffffffll;
}

[[nodiscard]] rund::kernel::i64 Lerp(const rund::kernel::i64 lhs,
                                     const rund::kernel::i64 rhs,
                                     const rund::kernel::i64 amount) noexcept {
  constexpr rund::kernel::i64 fixed_max = 0x7fffffffffffffffll;
  const rund::kernel::i64 t = Saturate(amount);
  return rund::math64::detail::ScalarAddSat(
      fixed::QuantizeProduct(lhs,
                             rund::math64::detail::ScalarSubSat(fixed_max, t)),
      fixed::QuantizeProduct(rhs, t));
}

[[nodiscard]] rund::kernel::i64
Bilerp(const rund::kernel::i64 x00, const rund::kernel::i64 x10,
       const rund::kernel::i64 x01, const rund::kernel::i64 x11,
       const rund::kernel::i64 tx, const rund::kernel::i64 ty) noexcept {
  return Lerp(Lerp(x00, x10, tx), Lerp(x01, x11, tx), ty);
}

[[nodiscard]] rund::kernel::i64 ScalarResult(const Work &work,
                                             const std::size_t index) noexcept {
  const bool ordered = ((work.lhs[index] < work.rhs[index]) ||
                        (work.lhs[index] <= work.rhs[index])) &&
                       (((work.lhs[index] == work.rhs[index]) ||
                         (work.lhs[index] != work.rhs[index])) ||
                        ((work.hi[index] > work.lo[index]) &&
                         (work.hi[index] >= work.lhs[index])));
  const rund::kernel::i64 clamped = rund::math64::detail::ScalarClamp(
      rund::math64::detail::ScalarAddWrap(
          rund::math64::detail::ScalarMin(work.lhs[index], work.rhs[index]),
          rund::math64::detail::ScalarMax(work.lhs[index], work.rhs[index])),
      work.lo[index], work.hi[index]);
  const rund::kernel::i64 unary = SaturatedSum(
      clamped, -static_cast<rund::math64::detail::i128>(work.lhs[index]),
      -static_cast<rund::math64::detail::i128>(work.rhs[index]),
      rund::math64::detail::ScalarAbs(work.lhs[index]),
      rund::math64::detail::ScalarAbsMagnitude(work.rhs[index]),
      rund::math64::detail::ScalarSign(work.rhs[index]));
  return !ordered ? unary : 0;
}

[[nodiscard]] rund::kernel::i64 BitResult(const Work &work,
                                          const std::size_t index) noexcept {
  const auto lhs_bits = std::bit_cast<rund::kernel::u64>(work.lhs[index]);
  const auto rhs_bits = std::bit_cast<rund::kernel::u64>(work.rhs[index]);
  return SaturatedSum(
      std::bit_cast<rund::kernel::i64>((lhs_bits & rhs_bits) |
                                       ((~lhs_bits) ^ (rhs_bits << 5u))),
      cpu::LogicalShiftRight64(work.lhs[index], 9u),
      cpu::ArithmeticShiftRight64(work.rhs[index], 11u));
}

[[nodiscard]] rund::kernel::i64
ArithmeticResult(const Work &work, const std::size_t index) noexcept {
  const auto lhs_bits = std::bit_cast<rund::kernel::u64>(work.lhs[index]);
  const auto rhs_bits = std::bit_cast<rund::kernel::u64>(work.rhs[index]);
  return SaturatedSum(
      rund::math64::detail::ScalarAddSat(work.lhs[index], work.rhs[index]),
      std::bit_cast<rund::kernel::i64>(
          rund::math64::detail::ScalarAddSatUnsigned(lhs_bits, rhs_bits)),
      rund::math64::detail::ScalarSubSat(work.lhs[index], work.rhs[index]),
      rund::math64::detail::ScalarNegPositiveFixed(work.positive[index]),
      fixed::QuantizeProduct(work.lhs[index], work.rhs[index]),
      fixed::QuantizeScaledProduct(
          work.lhs[index],
          std::bit_cast<rund::kernel::u64>(work.positive[index])),
      std::bit_cast<rund::kernel::i64>(fixed::QuantizeUnsignedProduct(
          std::bit_cast<rund::kernel::u64>(work.positive[index]), rhs_bits)),
      fixed::QuantizeMulAdd(work.lhs[index], work.rhs[index],
                            work.addend[index]),
      Saturate(work.positive[index]), Step(work.lhs[index], work.rhs[index]),
      Lerp(work.lhs[index], work.rhs[index], work.positive[index]),
      Bilerp(work.lhs[index], work.rhs[index], work.lo[index], work.hi[index],
             work.positive[index], work.positive[index]));
}

[[nodiscard]] rund::kernel::i64
NonlinearResult(const Work &work, const std::size_t index) noexcept {
  return SaturatedSum(fixed::DivNearestEven(work.lhs[index], work.rhs[index]),
                      fixed::RecipNearestEven(work.positive[index]),
                      fixed::SqrtFloor(work.positive[index]),
                      fixed::RsqrtNearestEven(work.positive[index]));
}

[[nodiscard]] rund::kernel::i64 ExpectedLane(const Work &work,
                                             const std::size_t index) noexcept {
  return SaturatedSum(ScalarResult(work, index), BitResult(work, index),
                      ArithmeticResult(work, index),
                      NonlinearResult(work, index));
}

} // namespace node_accel_contract::cpu::ops64
