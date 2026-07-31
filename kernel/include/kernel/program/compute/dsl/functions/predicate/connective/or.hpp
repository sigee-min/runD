#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
predicate_or(const ComputeValue lhs, const ComputeValue rhs) noexcept {
  return detail::Binary(rund::kernel::IrOp::PredicateOr, lhs, rhs);
}

template <detail::ConstantLiteral T>
[[nodiscard]] inline ComputeValue predicate_or(const ComputeValue lhs,
                                               const T rhs) noexcept {
  return predicate_or(lhs, detail::ConstantValue(lhs, rhs));
}

template <detail::ConstantLiteral T>
[[nodiscard]] inline ComputeValue
predicate_or(const T lhs, const ComputeValue rhs) noexcept {
  return predicate_or(detail::ConstantValue(rhs, lhs), rhs);
}

} // namespace rund::compute_dsl
