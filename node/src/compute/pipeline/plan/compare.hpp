#pragma once

#include "../state.hpp"

#include <algorithm>
#include <span>

namespace rund::compute::detail {

[[nodiscard]] inline bool
same_resolved_view(const PipelineResolvedViewPlan &left,
                   const PipelineResolvedViewPlan &right) noexcept {
  // Access-role and hidden-output markers do not alter the private Job View.
  // Every owner and byte/element coordinate does.
  return left.resource == right.resource &&
         left.declared_type == right.declared_type &&
         left.declared_format == right.declared_format &&
         left.declared_backing_bytes == right.declared_backing_bytes &&
         left.offset == right.offset && left.count == right.count &&
         left.stride == right.stride &&
         left.element_bytes == right.element_bytes &&
         left.alignment == right.alignment &&
         left.offset_bytes == right.offset_bytes &&
         left.stride_bytes == right.stride_bytes &&
         left.payload_bytes == right.payload_bytes &&
         left.span_bytes == right.span_bytes;
}

[[nodiscard]] inline bool same_resolved_views(
    const std::span<const PipelineResolvedViewPlan> left,
    const std::span<const PipelineResolvedViewPlan> right) noexcept {
  return left.size() == right.size() &&
         std::equal(left.begin(), left.end(), right.begin(),
                    same_resolved_view);
}

[[nodiscard]] inline bool same_recurrence_phase(
    const PipelineBuildStep &left,
    const PipelineStepResourcePlan &left_resources,
    const PipelineWindowControl *const left_control,
    const PipelineBuildStep &right,
    const PipelineStepResourcePlan &right_resources,
    const PipelineWindowControl *const right_control) noexcept {
  const bool same_control =
      (left_control == nullptr && right_control == nullptr) ||
      (left_control != nullptr && right_control != nullptr &&
       *left_control == *right_control);
  const bool same_outputs =
      left_resources.outputs.size() == right_resources.outputs.size() &&
      std::equal(left_resources.outputs.begin(), left_resources.outputs.end(),
                 right_resources.outputs.begin(),
                 [](const PipelineResolvedOutputPlan &lhs,
                    const PipelineResolvedOutputPlan &rhs) {
                   return lhs.physical == rhs.physical &&
                          same_resolved_view(lhs.view, rhs.view);
                 });
  return left.program == right.program &&
         left.logical_step == right.logical_step &&
         left.iteration_bound == right.iteration_bound &&
         left.nested == right.nested && left.route == right.route &&
         left.writes_each_iteration == right.writes_each_iteration &&
         same_control && same_outputs &&
         same_resolved_views(left_resources.inputs, right_resources.inputs);
}

} // namespace rund::compute::detail
