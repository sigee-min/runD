#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue absdiff(const ComputeValue lhs,
                                         const ComputeValue rhs) noexcept {
  return abs(sub_sat(lhs, rhs));
}

} // namespace rund::compute_dsl
