#pragma once

#include "../state.hpp"

#include <algorithm>
#include <span>

namespace rund::compute::detail {

[[nodiscard]] inline bool same_view(const PipelineBinding &left,
                                    const PipelineBinding &right) noexcept {
  const bool same_owner =
      left.owner == right.owner &&
      (left.owner != PipelineBinding::external || left.buffer == right.buffer);
  return same_owner && left.type == right.type && left.format == right.format &&
         left.offset == right.offset && left.count == right.count &&
         left.stride == right.stride &&
         left.element_bytes == right.element_bytes &&
         left.alignment == right.alignment &&
         left.backing_bytes == right.backing_bytes;
}

[[nodiscard]] inline bool
same_bindings(const std::span<const PipelineBinding> left,
              const std::span<const PipelineBinding> right) noexcept {
  return left.size() == right.size() &&
         std::equal(left.begin(), left.end(), right.begin(), same_view);
}

[[nodiscard]] inline bool
same_recurrence_phase(const PipelineBuildStep &left,
                      const PipelineBuildStep &right) noexcept {
  return left.program == right.program &&
         left.logical_step == right.logical_step &&
         left.iteration_bound == right.iteration_bound &&
         left.nested == right.nested && left.route == right.route &&
         left.writes_each_iteration == right.writes_each_iteration &&
         left.window_max == right.window_max &&
         left.window_tile == right.window_tile &&
         left.window_terminal == right.window_terminal &&
         left.window_expected == right.window_expected &&
         same_bindings(left.inputs, right.inputs) &&
         same_bindings(left.outputs, right.outputs);
}

} // namespace rund::compute::detail
