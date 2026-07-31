#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue centered(const CenteredOpAbs,
                                           const ComputeValue value,
                                           const ComputeValue center) noexcept {
  return abs(centered(value, center));
}

} // namespace rund::compute_dsl
