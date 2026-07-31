#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue add_sat(const ComputeValue lhs,
                                          const ComputeValue rhs) noexcept {
  return detail::Binary(rund::kernel::IrOp::AddSat, lhs, rhs);
}

[[nodiscard]] inline ComputeValue
add_sat_unsigned(const ComputeValue lhs, const ComputeValue rhs) noexcept {
  return detail::Binary(rund::kernel::IrOp::AddSatUnsigned, lhs, rhs);
}

[[nodiscard]] inline ComputeValue sub_sat(const ComputeValue lhs,
                                          const ComputeValue rhs) noexcept {
  return detail::Binary(rund::kernel::IrOp::SubSat, lhs, rhs);
}

[[nodiscard]] inline ComputeValue
neg_positive_fixed(const ComputeValue value) noexcept {
  return detail::Unary(rund::kernel::IrOp::NegPositiveFixed, value);
}

} // namespace rund::compute_dsl
