#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
activation(const ActivationOpHardSigmoid, const ComputeValue value) noexcept {
  return saturate(mean(value, fixed_one(value)));
}

[[nodiscard]] inline ComputeValue
activation(const ActivationOpHardSwish, const ComputeValue value) noexcept {
  return mul_fixed(value, activation(ActivationOp::HardSigmoid, value));
}

[[nodiscard]] inline ComputeValue
activation(const ActivationOpHardTanh, const ComputeValue value) noexcept {
  return clip(value, fixed_one(value));
}

} // namespace rund::compute_dsl
