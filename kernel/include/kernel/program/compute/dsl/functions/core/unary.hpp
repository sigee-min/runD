#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue neg(const ComputeValue value) noexcept {
  return detail::Unary(rund::kernel::IrOp::Neg, value);
}

[[nodiscard]] inline ComputeValue abs(const ComputeValue value) noexcept {
  return detail::Unary(rund::kernel::IrOp::Abs, value);
}

[[nodiscard]] inline ComputeValue
abs_magnitude(const ComputeValue value) noexcept {
  return detail::Unary(rund::kernel::IrOp::AbsMagnitude, value);
}

[[nodiscard]] inline ComputeValue sign(const ComputeValue value) noexcept {
  return detail::Unary(rund::kernel::IrOp::Sign, value);
}

} // namespace rund::compute_dsl
