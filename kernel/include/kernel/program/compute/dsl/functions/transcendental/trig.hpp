#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue sin(const ComputeValue value) noexcept {
  return detail::Unary(rund::kernel::IrOp::Sin, value);
}

[[nodiscard]] inline ComputeValue cos(const ComputeValue value) noexcept {
  return detail::Unary(rund::kernel::IrOp::Cos, value);
}

[[nodiscard]] inline ComputeValue tan(const ComputeValue value) noexcept {
  return detail::Unary(rund::kernel::IrOp::Tan, value);
}

} // namespace rund::compute_dsl
