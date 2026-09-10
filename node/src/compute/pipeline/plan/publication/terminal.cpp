#include "../../state/assembly.hpp"
#include "internal.hpp"

#include "../../output.hpp"
#include "../compare.hpp"

#include <kernel/program/compute/binding/model.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace rund::compute::detail::pipeline_publication_detail {

Result<PipelinePublicationPlan> plan_terminal_publication(
    const PipelineBuildState &build,
    const PipelineBuildTerminalPublication &publication,
    const std::span<const std::uint32_t> window_states,
    const std::span<const PipelineStepResourcePlan> step_resources,
    const std::span<const PipelineWindowControl> window_controls,
    PipelineScheduleResources &resources) {
  const PipelineBuildPublicationEdge &edge = publication.edge;
  auto base = resolve_publication_base(build, edge);
  const Location location =
      base ? publication_location(build, base->step) : Location{};
  if (!base || base->step.value >= window_states.size() ||
      window_states[base->step.value] == PipelineResourceUnassigned) {
    return Result<PipelinePublicationPlan>::fail(Reason::PipelineInvalid,
                                                 location);
  }
  const PipelineBuildOutputCoordinate base_coordinate{
      .step = base->step,
      .output = edge.output,
  };
  auto base_output = resolve_publication_output(build, step_resources,
                                                base_coordinate, location);
  if (!base_output || base->step.value >= step_resources.size() ||
      base_output->physical.value >=
          step_resources[base->step.value].inputs.size()) {
    return Result<PipelinePublicationPlan>::fail(
        base_output ? Reason::PipelineInvalid : base_output.reason(), location);
  }
  auto source_coordinate = resolve_publication_source(build, publication);
  auto selected_output =
      source_coordinate
          ? resolve_publication_output(build, step_resources,
                                       *source_coordinate, location)
          : Result<SealedPublicationOutput>::fail(source_coordinate.reason(),
                                                  location);
  if (!selected_output ||
      selected_output->physical.value != base_output->physical.value) {
    return Result<PipelinePublicationPlan>::fail(
        selected_output ? Reason::PipelineInvalid : selected_output.reason(),
        location);
  }
  auto target =
      seal_publication_view(resources, edge.target, ResourceAccess::Write,
                            rund::kernel::kResidentUsageWrite, location);
  if (!target) {
    return Result<PipelinePublicationPlan>::fail(target.reason(),
                                                 target.location());
  }
  const std::uint32_t state = window_states[base->step.value];
  if (state >= window_controls.size()) {
    return Result<PipelinePublicationPlan>::fail(Reason::PipelineInvalid,
                                                 location);
  }
  const PipelineWindowControl &control = window_controls[state];
  if (control.final < PipelineWindow::first ||
      control.final > PipelineWindow::second ||
      edge.target.owner != PipelineBinding::external) {
    return Result<PipelinePublicationPlan>::fail(Reason::PipelineInvalid,
                                                 location);
  }

  PipelineTerminalPublicationPlan planned{
      .target =
          PipelinePublicationTargetPlan{
              .view = *target,
          },
      .state = state,
      .output = base_output->physical,
  };
  const PipelineResolvedViewPlan *selected_bank = nullptr;
  for (std::size_t bank = 0u; bank < planned.sources.size(); ++bank) {
    const PipelineResolvedViewPlan *view = nullptr;
    if (bank == PipelineWindow::seed) {
      view =
          &step_resources[base->step.value].inputs[base_output->physical.value];
    } else {
      std::size_t step_index = base->step.value;
      if (bank == PipelineWindow::second &&
          step_index + 1u < build.steps.size() &&
          window_states[step_index + 1u] == planned.state) {
        ++step_index;
      }
      auto bank_output =
          resolve_publication_output(build, step_resources,
                                     PipelineBuildOutputCoordinate{
                                         .step = {.value = step_index},
                                         .output = edge.output,
                                     },
                                     location);
      if (!bank_output ||
          bank_output->physical.value != base_output->physical.value) {
        return Result<PipelinePublicationPlan>::fail(
            bank_output ? Reason::PipelineInvalid : bank_output.reason(),
            location);
      }
      view = &bank_output->output->view;
    }
    if (bank == control.final) {
      selected_bank = view;
    }
    auto source = resources.publication_view(
        *view, rund::kernel::kResidentUsageRead, location);
    if (!source) {
      return Result<PipelinePublicationPlan>::fail(source.reason(),
                                                   source.location());
    }
    planned.sources[bank] = *source;
  }
  if (selected_bank == nullptr || control.final >= planned.sources.size() ||
      !same_resolved_view(*selected_bank, selected_output->output->view)) {
    return Result<PipelinePublicationPlan>::fail(Reason::PipelineInvalid,
                                                 location);
  }
  for (const PipelinePublicationViewPlan &source : planned.sources) {
    if (source.type != target->type || source.format != target->format ||
        source.identity.count != target->identity.count ||
        source.identity.element_bytes != target->identity.element_bytes) {
      return Result<PipelinePublicationPlan>::fail(Reason::PipelineInvalid,
                                                   location);
    }
  }
  if (planned.sources[control.final].identity.resource_ordinal ==
          target->identity.resource_ordinal ||
      planned.sources[control.final].identity.offset_bytes != 0u ||
      planned.sources[control.final].identity.stride_bytes !=
          planned.sources[control.final].identity.element_bytes) {
    return Result<PipelinePublicationPlan>::fail(Reason::PipelineInvalid,
                                                 location);
  }
  if (!PipelineScheduleResources::append(
          resources.publication_accesses,
          planned.sources[control.final].identity, 0u) ||
      !PipelineScheduleResources::append(resources.publication_accesses,
                                         planned.target.view.identity, 0u)) {
    return Result<PipelinePublicationPlan>::fail(Reason::PipelineCapacity,
                                                 location);
  }
  PipelineResolvedResourcePlan &target_resource =
      resources.resources[target->identity.resource_ordinal];
  target_resource.output = true;
  target_resource.terminal_publish = true;
  target_resource.first_write =
      std::min(target_resource.first_write,
               static_cast<std::uint32_t>(
                   build.steps.empty() ? 0u : build.steps.size() - 1u));
  return Result<PipelinePublicationPlan>::success(std::move(planned));
}

} // namespace rund::compute::detail::pipeline_publication_detail
