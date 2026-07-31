#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
activation(const ActivationOpLeakyRelu, const ComputeValue value,
           const ComputeValue slope) noexcept {
  return select(is_nonneg(value), value, mul_fixed(value, slope));
}

} // namespace rund::compute_dsl
