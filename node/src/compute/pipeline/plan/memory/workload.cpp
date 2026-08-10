#include "local.hpp"

#include "../arena.hpp"
#include "../compare.hpp"
#include "../prepare.hpp"
#include "../resource.hpp"

#include "../../../../accel/kernel/recurrence.hpp"
#include "../../../backend.hpp"
#include "../../../buffer/local.hpp"
#include "../../../cpu/run/state.hpp"
#include "../../../job/local.hpp"
#include "../../../memory/arena.hpp"
#include "../../../status.hpp"
#include "../../../type.hpp"

#include <kernel/core/checked.hpp>
#include <rund/compute/abi/observe.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rund::compute::detail {

[[nodiscard]] Result<PipelinePlanningTotals>
project_pipeline_workload(const PipelineBuildState &build,
                          PipelineMemoryPlan &plan) {
  PipelinePlan &summary = plan.summary;
  std::uint64_t nested_commands = 0u;
  std::uint64_t logical_workspace = 0u;
  std::uint64_t live_workspace = 0u;
  std::size_t covered_until = 0u;
  for (std::size_t nested_index = 0u;
       nested_index < build.nested_windows.size(); ++nested_index) {
    const PipelineBuildNestedWindow &nested =
        build.nested_windows[nested_index];
    const node::accel::detail::NestedTemplateShape &shape = nested.shape;
    const bool state_range =
        shape.valid() && shape.first() < plan.window_states.size();
    const std::uint32_t state = state_range ? plan.window_states[shape.first()]
                                            : PipelineResourceUnassigned;
    const PipelineWindowControl *const control =
        state != PipelineResourceUnassigned &&
                state < plan.window_controls.size()
            ? &plan.window_controls[state]
            : nullptr;
    node::accel::detail::NestedTemplateShape expected_shape{};
    const bool window_shape =
        control != nullptr &&
        node::accel::detail::ProveNestedTemplateShape(
            shape.first(), control->maximum, control->tile, shape.inner_bound(),
            expected_shape) &&
        expected_shape == shape;
    if (nested_index >= std::numeric_limits<std::uint16_t>::max() ||
        !shape.valid() || shape.first() < covered_until ||
        shape.end() > build.steps.size() || !window_shape) {
      return Result<PipelinePlanningTotals>::fail(Reason::PipelineInvalid);
    }
    const auto nested_id = static_cast<std::uint16_t>(nested_index + 1u);
    std::uint64_t retained_entries = 0u;
    for (std::size_t index = shape.first(); index < shape.end(); ++index) {
      node::accel::detail::NestedTemplateRouteProjection route{};
      std::size_t retained_candidate = index;
      if (!shape.project(index, route) ||
          !shape.retained_owner(index, retained_candidate) ||
          retained_candidate > index) {
        return Result<PipelinePlanningTotals>::fail(Reason::PipelineInvalid);
      }
      const std::size_t phase_first =
          route.phase == node::accel::detail::NestedTemplatePhase::Seed
              ? shape.seed_first()
          : route.phase == node::accel::detail::NestedTemplatePhase::Action
              ? shape.action_first()
              : shape.fold_first();
      const PipelineBuildStep &step = build.steps[index];
      if (step.program != build.steps[phase_first].program ||
          step.nested != nested_id ||
          step.route != pipeline_route(route.phase) ||
          step.iteration != route.iteration ||
          step.iteration_bound != route.bound) {
        return Result<PipelinePlanningTotals>::fail(Reason::PipelineInvalid);
      }
      const std::size_t expected_owner =
          retained_candidate == index ? index
                                      : plan.job_owners[retained_candidate];
      if (plan.job_owners[index] != expected_owner) {
        return Result<PipelinePlanningTotals>::fail(Reason::PipelineInvalid);
      }
      retained_entries += expected_owner == index ? 1u : 0u;
    }
    if (retained_entries != shape.retained_entry_count()) {
      return Result<PipelinePlanningTotals>::fail(Reason::PipelineInvalid);
    }
    covered_until = shape.end();
    summary.outer_window_count =
        std::max(summary.outer_window_count,
                 static_cast<std::uint64_t>(shape.outer_bound()));
    summary.tile_capacity = std::max(summary.tile_capacity,
                                     static_cast<std::uint64_t>(control->tile));
    summary.inner_iteration_count =
        std::max(summary.inner_iteration_count,
                 static_cast<std::uint64_t>(shape.inner_bound()));
    if (!kernel::checked::add(nested_commands,
                              shape.authored_occurrence_count(),
                              nested_commands)) {
      return Result<PipelinePlanningTotals>::fail(Reason::PipelineCapacity);
    }
    const graph::MemoryPlan &seed_memory =
        build.steps[shape.seed_first()].program->graph_info.memory;
    const graph::MemoryPlan &fold_memory =
        build.steps[shape.fold_first()].program->graph_info.memory;
    const std::uint64_t action_logical =
        shape.action_count() == 0u
            ? 0u
            : build.steps[shape.action_first()]
                  .program->graph_info.memory.logical_bytes;
    std::uint64_t per_window = 0u;
    std::uint64_t action_total = 0u;
    std::uint64_t all_windows = 0u;
    if (!kernel::checked::mul(action_logical,
                              static_cast<std::uint64_t>(shape.action_count()),
                              action_total) ||
        !kernel::checked::add(seed_memory.logical_bytes, action_total,
                              per_window) ||
        !kernel::checked::add(per_window, fold_memory.logical_bytes,
                              per_window) ||
        !kernel::checked::mul(per_window,
                              static_cast<std::uint64_t>(shape.seed_count()),
                              all_windows) ||
        !kernel::checked::add(logical_workspace, all_windows,
                              logical_workspace)) {
      return Result<PipelinePlanningTotals>::fail(Reason::PipelineCapacity);
    }
  }
  std::size_t nested_index = 0u;
  for (std::size_t step_index = 0u; step_index < build.steps.size();
       ++step_index) {
    while (nested_index < build.nested_windows.size() &&
           step_index >= build.nested_windows[nested_index].shape.end()) {
      ++nested_index;
    }
    const bool nested =
        nested_index < build.nested_windows.size() &&
        step_index >= build.nested_windows[nested_index].shape.first();
    const PipelineBuildStep &step = build.steps[step_index];
    if (!nested) {
      if (step.nested != 0u || step.route != PipelineRoute::Ordinary) {
        return Result<PipelinePlanningTotals>::fail(Reason::PipelineInvalid);
      }
      if (!kernel::checked::add(logical_workspace,
                                step.program->graph_info.memory.logical_bytes,
                                logical_workspace)) {
        return Result<PipelinePlanningTotals>::fail(Reason::PipelineCapacity);
      }
    }
    live_workspace =
        std::max(live_workspace, step.program->graph_info.memory.live_bytes);
  }
  std::size_t internal_resource_count = 0u;
  for (const PipelineResolvedResourcePlan &resource : plan.resources) {
    if (const auto *external =
            std::get_if<PipelineExternalResourcePlan>(&resource.locator)) {
      if (external->owner == nullptr ||
          !kernel::checked::add(summary.persistent_bytes, resource.bytes,
                                summary.persistent_bytes)) {
        return Result<PipelinePlanningTotals>::fail(
            external->owner == nullptr ? Reason::PipelineInvalid
                                       : Reason::PipelineCapacity);
      }
    } else {
      ++internal_resource_count;
      if (!kernel::checked::add(summary.state_bytes, resource.bytes,
                                summary.state_bytes)) {
        return Result<PipelinePlanningTotals>::fail(Reason::PipelineCapacity);
      }
    }
  }
  if (internal_resource_count != build.internals.size()) {
    return Result<PipelinePlanningTotals>::fail(Reason::PipelineInvalid);
  }
  for (const PipelinePublicationPlan &publication : plan.publications) {
    const PipelinePublicationTargetPlan &target =
        pipeline_publication_target(publication);
    if (target.view.identity.resource_ordinal >= plan.resources.size() ||
        !std::holds_alternative<PipelineExternalResourcePlan>(
            plan.resources[target.view.identity.resource_ordinal].locator)) {
      return Result<PipelinePlanningTotals>::fail(Reason::PipelineInvalid);
    }
    std::uint64_t bytes = 0u;
    std::uint64_t elements = 0u;
    std::uint64_t element_bytes = 0u;
    std::uint64_t occurrences = 1u;
    const std::uint32_t state =
        std::visit([](const auto &typed) { return typed.state; }, publication);
    if (state >= plan.window_controls.size()) {
      return Result<PipelinePlanningTotals>::fail(Reason::PipelineInvalid);
    }
    const PipelineWindowControl &control = plan.window_controls[state];
    if (const auto *window =
            std::get_if<PipelineWindowPublicationPlan>(&publication)) {
      elements = control.maximum;
      element_bytes = window->source.identity.element_bytes;
      const node::accel::detail::NestedTemplateShape *const nested_shape =
          pipeline_build_nested_shape(build, state);
      if (nested_shape == nullptr || !nested_shape->valid()) {
        return Result<PipelinePlanningTotals>::fail(Reason::PipelineInvalid);
      }
      occurrences = nested_shape->outer_bound();
    } else {
      const auto &terminal =
          std::get<PipelineTerminalPublicationPlan>(publication);
      if (control.final < PipelineWindow::first ||
          control.final > PipelineWindow::second ||
          control.final >= terminal.sources.size()) {
        return Result<PipelinePlanningTotals>::fail(Reason::PipelineInvalid);
      }
      elements = terminal.sources[control.final].identity.count;
      element_bytes = terminal.sources[control.final].identity.element_bytes;
    }
    if (element_bytes == 0u ||
        !kernel::checked::mul(elements, element_bytes, bytes) ||
        !kernel::checked::add(summary.publish_bytes, bytes,
                              summary.publish_bytes) ||
        !kernel::checked::add(summary.publish_count, occurrences,
                              summary.publish_count)) {
      return Result<PipelinePlanningTotals>::fail(Reason::PipelineCapacity);
    }
  }
  return Result<PipelinePlanningTotals>::success(PipelinePlanningTotals{
      .nested_commands = nested_commands,
      .logical_workspace = logical_workspace,
      .live_workspace = live_workspace,
      .internal_resource_count = internal_resource_count,
  });
}

} // namespace rund::compute::detail
