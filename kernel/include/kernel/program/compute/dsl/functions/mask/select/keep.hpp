#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
keep_if(const ComputeValue value, const ComputeValue predicate) noexcept {
  return select(predicate, value, fixed_zero(value));
}

} // namespace rund::compute_dsl
