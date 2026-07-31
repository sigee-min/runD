#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
proportion(const ComputeValue value, const ComputeValue total) noexcept {
  return saturate(ratio(value, total));
}

} // namespace rund::compute_dsl
