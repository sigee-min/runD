#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
activation(const ActivationOpRelu, const ComputeValue value) noexcept {
  return positive_part(value);
}

[[nodiscard]] inline ComputeValue
activation(const ActivationOpRelu, const ComputeValue value,
           const ComputeValue upper) noexcept {
  return min(positive_part(value), positive_part(upper));
}

} // namespace rund::compute_dsl
