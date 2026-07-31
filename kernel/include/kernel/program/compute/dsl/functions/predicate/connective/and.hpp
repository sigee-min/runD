#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
predicate_and(const ComputeValue lhs, const ComputeValue rhs) noexcept {
  return detail::Binary(rund::kernel::IrOp::PredicateAnd, lhs, rhs);
}

template <detail::ConstantLiteral T>
[[nodiscard]] inline ComputeValue predicate_and(const ComputeValue lhs,
                                                const T rhs) noexcept {
  return predicate_and(lhs, detail::ConstantValue(lhs, rhs));
}

template <detail::ConstantLiteral T>
[[nodiscard]] inline ComputeValue
predicate_and(const T lhs, const ComputeValue rhs) noexcept {
  return predicate_and(detail::ConstantValue(rhs, lhs), rhs);
}

} // namespace rund::compute_dsl
