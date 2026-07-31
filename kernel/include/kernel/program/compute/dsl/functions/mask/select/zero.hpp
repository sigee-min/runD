#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
zero_if(const ComputeValue value, const ComputeValue predicate) noexcept {
  return select(predicate, fixed_zero(value), value);
}

} // namespace rund::compute_dsl
