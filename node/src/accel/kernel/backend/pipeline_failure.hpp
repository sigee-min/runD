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
    const BackendWindow *const window = entry.recurrence.window;
    if (window == nullptr ||
        !window->valid_occurrence(entry.transduced_action())) {
      clear_window();
      return;
    }
    window_route(window);
  }

  // A compact Seed template owns one exact outer coordinate. Action and Fold
  // templates are reused across outer iterations, so their placeholder outer
  // value must not be promoted to public failure evidence before expansion.
  void compact_template_route(const std::uint32_t template_index,
                              const BackendRecurrence &recurrence) noexcept {
    template_route(template_index);
    const BackendWindow *const window = recurrence.window;
    if (window != nullptr && window->phase == BackendWindowPhase::NestedSeed &&
        window->valid_occurrence(false)) {
      window_route(window);
    }
  }

  void node_route(const BackendBatchEntry &entry,
                  const std::size_t step_index) noexcept {
    occurrence_route(entry);
    node_for(entry.run, step_index);
  }

  void template_node_route(const BackendBatchEntry &entry,
                           const std::size_t step_index) noexcept {
    template_route(entry.template_index);
    node_for(entry.run, step_index);
  }

  // A backend-owned status row may outlive the private BoundStep cursor that
  // described it. Preserve the admitted graph-node coordinate in that row and
  // restore it without fabricating an occurrence or nested coordinate.
  void template_node_route(const std::uint32_t template_index,
                           const std::uint32_t node) noexcept {
    template_route(template_index);
    node_ = node;
  }

  void compact_template_node_route(const std::uint32_t template_index,
                                   const BackendRecurrence &recurrence,
                                   const BackendRun &run,
                                   const std::size_t step_index) noexcept {
    compact_template_route(template_index, recurrence);
    node_for(&run, step_index);
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
    rund::compute::PipelineNestedPhase phase{};
    if (!window->nested_phase(phase)) {
      return;
    }
    nested_phase_ = phase;
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

  void node_for(const BackendRun *const run,
                const std::size_t step_index) noexcept {
    if (run == nullptr || run->steps == nullptr ||
        step_index >= run->step_count) {
      return;
    }
    const BoundStep &step = run->steps[step_index];
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
