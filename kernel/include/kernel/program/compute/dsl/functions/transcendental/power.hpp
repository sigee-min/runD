#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue pow(const ComputeValue lhs,
                                      const ComputeValue rhs) noexcept {
  return exp(mul_fixed(rhs, log(lhs)));
}

} // namespace rund::compute_dsl
