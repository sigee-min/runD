#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue step(const ComputeValue edge,
                                       const ComputeValue value) noexcept {
  return select(lt(value, edge), fixed_zero(value), fixed_one(value));
}

} // namespace rund::compute_dsl
