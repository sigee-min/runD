#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
predicate_not(const ComputeValue value) noexcept {
  return detail::Unary(rund::kernel::IrOp::PredicateNot, value);
}

} // namespace rund::compute_dsl
