#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue centered(const ComputeValue value,
                                           const ComputeValue center) noexcept {
  return sub_sat(value, center);
}

} // namespace rund::compute_dsl
