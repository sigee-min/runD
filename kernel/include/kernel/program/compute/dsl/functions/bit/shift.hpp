#pragma once

namespace rund::compute_dsl {

template <rund::kernel::u32 Amount>
[[nodiscard]] inline ComputeValue shl_const(const ComputeValue value) noexcept {
  return detail::ConstShift(rund::kernel::IrOp::ShlConst, value, Amount);
}

template <rund::kernel::u32 Amount>
[[nodiscard]] inline ComputeValue
shr_logical_const(const ComputeValue value) noexcept {
  return detail::ConstShift(rund::kernel::IrOp::ShrLogicalConst, value,
                            Amount);
}

template <rund::kernel::u32 Amount>
[[nodiscard]] inline ComputeValue
shr_arithmetic_const(const ComputeValue value) noexcept {
  return detail::ConstShift(rund::kernel::IrOp::ShrArithmeticConst,
                            value, Amount);
}

} // namespace rund::compute_dsl
