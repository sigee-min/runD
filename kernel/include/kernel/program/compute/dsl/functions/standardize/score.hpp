#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
zscore(const ComputeValue value, const ComputeValue center,
       const ComputeValue scale) noexcept {
  return ratio(centered(value, center), scale);
}

} // namespace rund::compute_dsl
