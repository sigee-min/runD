#pragma once

#include "../prepared/failure.hpp"
#include "run.hpp"

#include <cstddef>
#include <cstdint>

namespace rund::node::accel::detail {

// Mutable preparation cursor used only while a backend is building a frozen
// pipeline.  Moving to a new stage clears route state, which prevents an old
// coordinate from being attached to an unrelated later failure.
class PreparedPipelineFailureContext final {
public:
  void stage(const PreparedPipelineFailureStage value) noexcept {
    stage_ = value;
    clear_route();
  }

  void template_route(const std::uint32_t template_index) noexcept {
    template_index_ = template_index;
    occurrence_index_ = PreparedPipelineUnknownCoordinate;
    node_ = PreparedPipelineUnknownCoordinate;
    clear_window();
  }

  void occurrence_route(const BackendBatchEntry &entry) noexcept {
    template_index_ = entry.template_index;
    occurrence_index_ = entry.occurrence_index;
    node_ = PreparedPipelineUnknownCoordinate;
    window_route(entry.recurrence.window);
  }

  // Private route materialization has not produced a physical occurrence yet,
  // but its authored recurrence already owns exact nested coordinates.
  void template_recurrence_route(const std::uint32_t template_index,
                                 const BackendRecurrence &recurrence) noexcept {
    template_route(template_index);
    window_route(recurrence.window);
  }

  void node_route(const BackendBatchEntry &entry,
                  const std::size_t step_index) noexcept {
    occurrence_route(entry);
    node_for(entry, step_index);
  }

  void template_node_route(const BackendBatchEntry &entry,
                           const std::size_t step_index) noexcept {
    template_route(entry.template_index);
    node_for(entry, step_index);
  }

  // A backend-owned status row may outlive the private BoundStep cursor that
  // described it. Preserve the admitted graph-node coordinate in that row and
  // restore it without fabricating an occurrence or nested coordinate.
  void template_node_route(const std::uint32_t template_index,
                           const std::uint32_t node) noexcept {
    template_route(template_index);
    node_ = node;
  }

  void template_node_recurrence_route(const std::uint32_t template_index,
                                      const BackendRecurrence &recurrence,
                                      const BackendRun &run,
                                      const std::size_t step_index) noexcept {
    template_recurrence_route(template_index, recurrence);
    if (run.steps != nullptr && step_index < run.step_count &&
        run.steps[step_index].step != nullptr) {
      node_ = run.steps[step_index].step->source.begin.index;
    }
  }

  void clear_route() noexcept {
    template_index_ = PreparedPipelineUnknownCoordinate;
    occurrence_index_ = PreparedPipelineUnknownCoordinate;
    node_ = PreparedPipelineUnknownCoordinate;
    clear_window();
  }

  [[nodiscard]] PreparedPipelineFailure
  failure(const char *const native_reason_key) const noexcept {
    return PreparedPipelineFailure{
        .stage = stage_,
        .template_index = template_index_,
        .occurrence_index = occurrence_index_,
        .node = node_,
        .outer_iteration = outer_iteration_,
        .inner_iteration = inner_iteration_,
        .nested_phase = nested_phase_,
        .native_reason_key =
            native_reason_key == nullptr || native_reason_key[0] == '\0'
                ? "accel_kernel_pipeline_invalid"
                : native_reason_key,
    };
  }

private:
  void window_route(const BackendWindow *const window) noexcept {
    clear_window();
    if (window == nullptr) {
      return;
    }
    nested_phase_ = window->nested_phase();
    if (rund::compute::pipeline_nested_phase_has_outer(nested_phase_)) {
      outer_iteration_ = window->outer_iteration;
    }
    if (rund::compute::pipeline_nested_phase_has_inner(nested_phase_)) {
      inner_iteration_ = window->inner_iteration;
    }
  }

  void clear_window() noexcept {
    outer_iteration_ = PreparedPipelineUnknownCoordinate;
    inner_iteration_ = PreparedPipelineUnknownCoordinate;
    nested_phase_ = rund::compute::PipelineNestedPhase::None;
  }

  void node_for(const BackendBatchEntry &entry,
                const std::size_t step_index) noexcept {
    if (entry.run == nullptr || entry.run->steps == nullptr ||
        step_index >= entry.run->step_count) {
      return;
    }
    const BoundStep &step = entry.run->steps[step_index];
    if (step.step != nullptr) {
      node_ = step.step->source.begin.index;
    }
  }
  PreparedPipelineFailureStage stage_{PreparedPipelineFailureStage::Unknown};
  std::uint32_t template_index_{PreparedPipelineUnknownCoordinate};
  std::uint32_t occurrence_index_{PreparedPipelineUnknownCoordinate};
  std::uint32_t node_{PreparedPipelineUnknownCoordinate};
  std::uint32_t outer_iteration_{PreparedPipelineUnknownCoordinate};
  std::uint32_t inner_iteration_{PreparedPipelineUnknownCoordinate};
  rund::compute::PipelineNestedPhase nested_phase_{
      rund::compute::PipelineNestedPhase::None};
};

} // namespace rund::node::accel::detail
