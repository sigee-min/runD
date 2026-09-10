#include "../../../state/assembly.hpp"
#include "local.hpp"

#include "../../resource.hpp"

#include "../../../../../accel/kernel/recurrence.hpp"
#include "../../../../status.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace rund::compute::detail {

Status collect_pipeline_accel_occurrences(
    const PipelineBuildState &build, PipelineMemoryPlan &plan,
    PipelineAccelPreparationDraft &draft) {
  const std::size_t size = build.steps.size();
  draft.entry_counts.assign(size, 0u);
  draft.occurrence_counts.assign(size, 0u);
  draft.window_counts.assign(size, 0u);
  draft.nested_group_counts.assign(size, 0u);
  draft.map_recurrence_group_counts.assign(size, 0u);
  draft.map_recurrence_history_group_counts.assign(size, 0u);
  draft.recurrence_hi.assign(size, 0u);
  draft.recurrence_lo.assign(size, 0u);
  draft.active_window_states.assign(size, 0u);
  draft.window_state_count = 0u;
  draft.window_descriptor_state_count = 0u;

  auto &entry_counts = draft.entry_counts;
  auto &occurrence_counts = draft.occurrence_counts;
  auto &window_counts = draft.window_counts;
  auto &nested_group_counts = draft.nested_group_counts;
  auto &map_recurrence_group_counts = draft.map_recurrence_group_counts;
  auto &map_recurrence_history_group_counts =
      draft.map_recurrence_history_group_counts;
  auto &recurrence_hi = draft.recurrence_hi;
  auto &recurrence_lo = draft.recurrence_lo;
  auto &active_window_states = draft.active_window_states;
  auto &window_state_count = draft.window_state_count;
  auto &window_descriptor_state_count = draft.window_descriptor_state_count;

  for (std::size_t index = 0u; index < build.steps.size(); ++index) {
    node::accel::detail::SeedPreparedKernelRecurrenceFingerprint(
        recurrence_hi[index], recurrence_lo[index]);
  }
  for (std::size_t index = 0u; index < build.steps.size(); ++index) {
    const PipelineBuildStep &step = build.steps[index];
    if (step.program == nullptr || step.program->empty()) {
      continue;
    }
    const std::size_t owner = plan.job_owners[index];
    if (owner >= build.steps.size() || owner > index) {
      return Status::fail(Reason::PipelineInvalid);
    }
    node::accel::detail::PreparedKernelRecurrenceIdentity identity{
        .logical_step = step.logical_step,
        .iteration = step.iteration,
        .bound = step.iteration_bound,
        .writes_each_iteration = step.writes_each_iteration,
    };
    constexpr std::uint32_t unassigned =
        std::numeric_limits<std::uint32_t>::max();
    const std::uint32_t state = plan.window_states[index];
    const PipelineWindowControl *control = nullptr;
    if (state != unassigned) {
      if (state >= plan.window_controls.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      control = &plan.window_controls[state];
    }
    std::uint64_t occurrences = 1u;
    if (step.nested != 0u) {
      const std::size_t nested_index = step.nested - 1u;
      if (nested_index >= build.nested_windows.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const PipelineBuildNestedWindow &nested =
          build.nested_windows[nested_index];
      node::accel::detail::NestedTemplateRouteProjection route{};
      if (control == nullptr || !nested.shape.valid() ||
          !nested.shape.project(index, route) ||
          !node::accel::detail::ProjectNestedRecurrenceIdentity(
              nested.shape, index,
              node::accel::detail::NestedTemplateRecurrenceIdentityBase{
                  .logical_step = step.logical_step,
                  .maximum = control->maximum,
                  .tile = control->tile,
                  .expected = control->expected,
                  .state = state,
                  .has_terminal = control->terminal !=
                                  std::numeric_limits<std::uint32_t>::max(),
              },
              identity) ||
          step.route != pipeline_route(route.phase) ||
          step.iteration != identity.iteration ||
          step.iteration_bound != identity.bound ||
          step.writes_each_iteration) {
        return Status::fail(Reason::PipelineInvalid);
      }
      occurrences = route.occurrence_count;
    } else if (control != nullptr) {
      identity.maximum = control->maximum;
      identity.tile = control->tile;
      identity.expected = control->expected;
      identity.outer_iteration = step.iteration;
      identity.outer_bound = step.iteration_bound;
      identity.inner_bound = 1u;
      identity.phase = node::accel::detail::BackendWindowPhase::Ordinary;
      identity.has_window = true;
      identity.has_terminal =
          control->terminal != std::numeric_limits<std::uint32_t>::max();
    }
    if (identity.has_window) {
      if (state == unassigned || state >= active_window_states.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      identity.state = state;
      window_state_count = std::max(window_state_count,
                                    static_cast<std::uint64_t>(state) + 1u);
      if (active_window_states[state] == 0u) {
        active_window_states[state] = 1u;
        ++window_descriptor_state_count;
      }
    }
    if (!node::accel::detail::MixPreparedKernelRecurrenceFingerprint(
            recurrence_hi[owner], recurrence_lo[owner], identity)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    if (!kernel::checked::add(entry_counts[owner], 1u, entry_counts[owner]) ||
        !kernel::checked::add(occurrence_counts[owner], occurrences,
                              occurrence_counts[owner]) ||
        (identity.has_window &&
         !kernel::checked::add(window_counts[owner], occurrences,
                               window_counts[owner]))) {
      return Status::fail(Reason::PipelineCapacity);
    }
  }
  for (const PipelineBuildNestedWindow &nested : build.nested_windows) {
    const node::accel::detail::NestedTemplateShape &shape = nested.shape;
    if (!shape.valid() || shape.seed_first() >= build.steps.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::size_t owner = plan.job_owners[shape.seed_first()];
    if (owner >= build.steps.size() ||
        build.steps[shape.seed_first()].program == nullptr ||
        build.steps[shape.seed_first()].program->empty()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    if (!kernel::checked::add(nested_group_counts[owner], 1u,
                              nested_group_counts[owner])) {
      return Status::fail(Reason::PipelineCapacity);
    }
    if (shape.action_group_candidate()) {
      if (shape.action_first() >= build.steps.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const std::size_t action_owner = plan.job_owners[shape.action_first()];
      if (action_owner >= build.steps.size() ||
          !kernel::checked::add(map_recurrence_group_counts[action_owner], 1u,
                                map_recurrence_group_counts[action_owner])) {
        return Status::fail(action_owner >= build.steps.size()
                                ? Reason::PipelineInvalid
                                : Reason::PipelineCapacity);
      }
    }
  }

  // A top-level Map recurrence is a whole-command-stream transform. It is
  // therefore one candidate group at most, never one group per iteration or
  // parity route. Runtime proves bindings/artifact eligibility against this
  // same authored marker before materializing any native recurrence owner.
  bool top_level_recurrence =
      build.steps.size() > 1u &&
      build.steps.size() <= std::numeric_limits<std::uint32_t>::max();
  const PipelineBuildStep *const top =
      top_level_recurrence ? &build.steps.front() : nullptr;
  for (std::size_t index = 0u;
       top_level_recurrence && index < build.steps.size(); ++index) {
    const PipelineBuildStep &step = build.steps[index];
    top_level_recurrence =
        step.program == top->program &&
        step.logical_step == top->logical_step && step.iteration == index &&
        step.iteration_bound == build.steps.size() && step.nested == 0u &&
        step.route == PipelineRoute::Ordinary &&
        plan.window_states[index] == PipelineResourceUnassigned &&
        step.writes_each_iteration == top->writes_each_iteration;
  }
  if (top_level_recurrence) {
    const std::size_t owner = plan.job_owners.front();
    if (owner >= build.steps.size() ||
        !kernel::checked::add(map_recurrence_group_counts[owner], 1u,
                              map_recurrence_group_counts[owner]) ||
        (top->writes_each_iteration &&
         !kernel::checked::add(map_recurrence_history_group_counts[owner], 1u,
                               map_recurrence_history_group_counts[owner]))) {
      return Status::fail(owner >= build.steps.size()
                              ? Reason::PipelineInvalid
                              : Reason::PipelineCapacity);
    }
  }

  return Status::success();
}

} // namespace rund::compute::detail
