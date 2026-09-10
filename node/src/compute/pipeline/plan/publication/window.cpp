#include "../../state/assembly.hpp"
#include "internal.hpp"

#include "../../output.hpp"
#include "../compare.hpp"

#include <kernel/program/compute/binding/model.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace rund::compute::detail::pipeline_publication_detail {

Result<PipelinePublicationPlan> plan_window_publication(
    const PipelineBuildState &build,
    const PipelineBuildWindowPublication &publication,
    const std::span<const std::uint32_t> window_states,
    const std::span<const PipelineStepResourcePlan> step_resources,
    const std::span<const PipelineWindowControl> window_controls,
    PipelineScheduleResources &resources) {
  const PipelineBuildPublicationEdge &edge = publication.edge;
  auto base = resolve_publication_base(build, edge);
  const Location location =
      base ? publication_location(build, base->step) : Location{};
  if (!base || base->step.value >= window_states.size() ||
      window_states[base->step.value] == PipelineResourceUnassigned ||
      window_states[base->step.value] >= window_controls.size()) {
    return Result<PipelinePublicationPlan>::fail(Reason::PipelineInvalid,
                                                 location);
  }
  const PipelineBuildStep &first = build.steps[base->step.value];
  auto source_coordinate = resolve_publication_source(build, publication);
  if (!source_coordinate) {
    return Result<PipelinePublicationPlan>::fail(source_coordinate.reason(),
                                                 location);
  }
  auto source_output = resolve_publication_output(build, step_resources,
                                                  *source_coordinate, location);
  if (!source_output) {
    return Result<PipelinePublicationPlan>::fail(source_output.reason(),
                                                 source_output.location());
  }
  auto source = resources.publication_view(
      source_output->output->view, rund::kernel::kResidentUsageRead, location);
  auto target =
      seal_publication_view(resources, edge.target, ResourceAccess::Write,
                            rund::kernel::kResidentUsageWrite, location);
  if (!source || !target) {
    const auto &failed = !source ? source : target;
    return Result<PipelinePublicationPlan>::fail(failed.reason(),
                                                 failed.location());
  }
  const std::uint32_t state = window_states[base->step.value];
  const PipelineWindowControl &control = window_controls[state];
  if (first.route != PipelineRoute::NestedFold || first.nested == 0u ||
      edge.target.owner != PipelineBinding::external ||
      source->type != target->type || source->format != target->format ||
      source->identity.resource_ordinal == target->identity.resource_ordinal ||
      source->identity.offset_bytes != 0u ||
      source->identity.stride_bytes != source->identity.element_bytes ||
      target->identity.stride_bytes != target->identity.element_bytes ||
      source->identity.count != control.tile ||
      target->identity.count != control.maximum ||
      source->identity.element_bytes != target->identity.element_bytes) {
    return Result<PipelinePublicationPlan>::fail(Reason::PipelineInvalid,
                                                 location);
  }
  const std::size_t nested_index = first.nested - 1u;
  if (nested_index >= build.nested_windows.size()) {
    return Result<PipelinePublicationPlan>::fail(Reason::PipelineInvalid,
                                                 location);
  }
  const PipelineBuildNestedWindow &nested = build.nested_windows[nested_index];
  if (!nested.shape.valid() || base->step.value != nested.shape.fold_first()) {
    return Result<PipelinePublicationPlan>::fail(Reason::PipelineInvalid,
                                                 location);
  }

  for (std::size_t route = 0u; route < nested.shape.fold_count(); ++route) {
    const PipelineBuildOutputCoordinate route_coordinate{
        .step = {.value = base->step.value + route},
        .output = edge.output,
    };
    auto route_output = resolve_publication_output(build, step_resources,
                                                   route_coordinate, location);
    if (!route_output ||
        route_output->physical.value != source_output->physical.value ||
        !same_resolved_view(route_output->output->view,
                            source_output->output->view)) {
      return Result<PipelinePublicationPlan>::fail(
          route_output ? Reason::PipelineInvalid : route_output.reason(),
          location);
    }
  }

  PipelineWindowPublicationPlan planned{
      .source = *source,
      .target =
          PipelinePublicationTargetPlan{
              .view = *target,
          },
      .state = state,
      .output = source_output->physical,
  };
  if (!PipelineScheduleResources::append(resources.publication_accesses,
                                         planned.source.identity, 0u) ||
      !PipelineScheduleResources::append(resources.publication_accesses,
                                         planned.target.view.identity, 0u) ||
      !PipelineScheduleResources::append(resources.publication_accesses,
                                         control.count.identity, 0u)) {
    return Result<PipelinePublicationPlan>::fail(Reason::PipelineCapacity,
                                                 location);
  }
  for (std::size_t route = 0u; route < nested.shape.fold_count(); ++route) {
    const std::size_t step_index = base->step.value + route;
    if (step_index >= build.steps.size() ||
        build.steps[step_index].route != PipelineRoute::NestedFold ||
        !PipelineScheduleResources::append(
            resources.accesses, planned.target.view.identity,
            static_cast<std::uint32_t>(step_index))) {
      return Result<PipelinePublicationPlan>::fail(Reason::PipelineInvalid,
                                                   location);
    }
  }
  PipelineResolvedResourcePlan &target_resource =
      resources.resources[target->identity.resource_ordinal];
  target_resource.output = true;
  target_resource.first_write =
      std::min(target_resource.first_write,
               static_cast<std::uint32_t>(nested.shape.first()));
  return Result<PipelinePublicationPlan>::success(std::move(planned));
}

} // namespace rund::compute::detail::pipeline_publication_detail
