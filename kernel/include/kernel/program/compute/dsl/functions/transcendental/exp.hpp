#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue exp(const ComputeValue value) noexcept {
  return detail::Unary(rund::kernel::IrOp::Exp, value);
}

} // namespace rund::compute_dsl
