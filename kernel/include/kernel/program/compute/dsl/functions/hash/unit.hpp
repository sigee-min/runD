#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
hash(const HashOpUnit, const ComputeValue value) noexcept {
  return bit_and(hash(value), detail::FixedFractionMaskConstant(value));
}

[[nodiscard]] inline ComputeValue
hash(const HashOpUnit, const ComputeValue value,
     const ComputeValue seed) noexcept {
  return bit_and(hash(value, seed), detail::FixedFractionMaskConstant(value));
}

} // namespace rund::compute_dsl
