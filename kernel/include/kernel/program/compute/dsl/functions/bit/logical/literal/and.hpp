#pragma once

namespace rund::compute_dsl {

template <detail::ConstantLiteral T>
[[nodiscard]] inline ComputeValue bit_and(const ComputeValue lhs,
                                          const T rhs) noexcept {
  return bit_and(lhs, detail::ConstantValue(lhs, rhs));
}

template <detail::ConstantLiteral T>
[[nodiscard]] inline ComputeValue bit_and(const T lhs,
                                          const ComputeValue rhs) noexcept {
  return bit_and(detail::ConstantValue(rhs, lhs), rhs);
}

} // namespace rund::compute_dsl
