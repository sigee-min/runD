#pragma once

#include "evaluation.hpp"

#include <cstdint>

namespace rund::node::accel::detail {
namespace range_plan_detail {

[[nodiscard]] constexpr bool LessOrEqual(const RangeCost &left,
                                         const RangeCost &right) noexcept {
  return left.global_read_bytes <= right.global_read_bytes &&
         left.global_write_bytes <= right.global_write_bytes &&
         left.workgroup_count <= right.workgroup_count &&
         left.combine_ops <= right.combine_ops &&
         left.inverse_ops <= right.inverse_ops &&
         left.scale_ops <= right.scale_ops &&
         left.shared_bytes <= right.shared_bytes &&
         left.scratch_bytes <= right.scratch_bytes &&
         left.dispatch_count <= right.dispatch_count &&
         left.launched_lanes <= right.launched_lanes;
}

[[nodiscard]] constexpr bool Dominates(const RangeCost &left,
                                       const RangeCost &right) noexcept {
  return LessOrEqual(left, right) && !(left == right);
}

template <typename T>
[[nodiscard]] constexpr int CompareScalar(const T left,
                                          const T right) noexcept {
  return left < right ? -1 : (right < left ? 1 : 0);
}

[[nodiscard]] constexpr int
CompareLexicographic(const CandidateEvaluation &left,
                     const CandidateEvaluation &right) noexcept {
#define RUND_RANGE_COMPARE(field)                                              \
  if (const int order = CompareScalar(left.cost.field, right.cost.field);      \
      order != 0) {                                                            \
    return order;                                                              \
  }
  RUND_RANGE_COMPARE(global_read_bytes)
  RUND_RANGE_COMPARE(global_write_bytes)
  RUND_RANGE_COMPARE(workgroup_count)
  RUND_RANGE_COMPARE(combine_ops)
  RUND_RANGE_COMPARE(inverse_ops)
  RUND_RANGE_COMPARE(scale_ops)
  RUND_RANGE_COMPARE(scratch_bytes)
  RUND_RANGE_COMPARE(dispatch_count)
  RUND_RANGE_COMPARE(shared_bytes)
  RUND_RANGE_COMPARE(launched_lanes)
#undef RUND_RANGE_COMPARE
  if (const int order = CompareScalar(
          static_cast<std::uint8_t>(left.candidate.disposition()),
          static_cast<std::uint8_t>(right.candidate.disposition()));
      order != 0) {
    return order;
  }
  if (const int order =
          CompareScalar(left.candidate.width(), right.candidate.width());
      order != 0) {
    return order;
  }
  return CompareScalar(left.candidate.radius_capacity(),
                       right.candidate.radius_capacity());
}

} // namespace range_plan_detail
} // namespace rund::node::accel::detail
