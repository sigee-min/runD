#pragma once

namespace rund::compute_dsl {

template <detail::ConstantLiteral T>
[[nodiscard]] inline ComputeValue bit_or(const ComputeValue lhs,
                                         const T rhs) noexcept {
  return bit_or(lhs, detail::ConstantValue(lhs, rhs));
}

template <detail::ConstantLiteral T>
[[nodiscard]] inline ComputeValue bit_or(const T lhs,
                                         const ComputeValue rhs) noexcept {
  return bit_or(detail::ConstantValue(rhs, lhs), rhs);
}

} // namespace rund::compute_dsl
