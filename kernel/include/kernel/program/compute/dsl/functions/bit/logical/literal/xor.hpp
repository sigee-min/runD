#pragma once

namespace rund::compute_dsl {

template <detail::ConstantLiteral T>
[[nodiscard]] inline ComputeValue bit_xor(const ComputeValue lhs,
                                          const T rhs) noexcept {
  return bit_xor(lhs, detail::ConstantValue(lhs, rhs));
}

template <detail::ConstantLiteral T>
[[nodiscard]] inline ComputeValue bit_xor(const T lhs,
                                          const ComputeValue rhs) noexcept {
  return bit_xor(detail::ConstantValue(rhs, lhs), rhs);
}

} // namespace rund::compute_dsl
