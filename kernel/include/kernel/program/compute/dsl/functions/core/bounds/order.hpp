#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue min(const ComputeValue lhs,
                                      const ComputeValue rhs) noexcept {
  return detail::Binary(rund::kernel::IrOp::Min, lhs, rhs);
}

[[nodiscard]] inline ComputeValue max(const ComputeValue lhs,
                                      const ComputeValue rhs) noexcept {
  return detail::Binary(rund::kernel::IrOp::Max, lhs, rhs);
}

} // namespace rund::compute_dsl
