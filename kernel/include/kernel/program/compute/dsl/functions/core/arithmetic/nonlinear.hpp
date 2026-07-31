#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue div_fixed(const ComputeValue lhs,
                                            const ComputeValue rhs) noexcept {
  return detail::Binary(rund::kernel::IrOp::DivFixed, lhs, rhs);
}

[[nodiscard]] inline ComputeValue recip(const ComputeValue value) noexcept {
  return detail::Unary(rund::kernel::IrOp::Recip, value);
}

[[nodiscard]] inline ComputeValue sqrt(const ComputeValue value) noexcept {
  return detail::Unary(rund::kernel::IrOp::Sqrt, value);
}

[[nodiscard]] inline ComputeValue rsqrt(const ComputeValue value) noexcept {
  return detail::Unary(rund::kernel::IrOp::Rsqrt, value);
}

} // namespace rund::compute_dsl
