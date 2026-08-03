#include "publication.hpp"

#include "../../size.hpp"

#include <kernel/program/compute/binding/model.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace rund::compute::detail {

namespace {

[[nodiscard]] bool publication_view_identity(
    const PipelineBinding &binding, const std::uint32_t ordinal,
    const std::uint32_t usage,
    node::accel::detail::PreparedKernelPublicationViewIdentity &out) noexcept {
  std::size_t offset_bytes = 0u;
  std::size_t stride_bytes = 0u;
  if (binding.element_bytes == 0u || binding.stride == 0u ||
      !size::multiply(binding.offset, binding.element_bytes, offset_bytes) ||
      !size::multiply(binding.stride, binding.element_bytes, stride_bytes)) {
    return false;
  }
  out = node::accel::detail::PreparedKernelPublicationViewIdentity{
      .backing_bytes = binding.backing_bytes,
      .offset_bytes = offset_bytes,
      .count = binding.count,
      .stride_bytes = stride_bytes,
      .element_bytes = binding.element_bytes,
      .resource_ordinal = ordinal,
      .usage = usage,
  };
  return true;
}

[[nodiscard]] constexpr PipelineNestedPhase
publication_phase(const PipelineRoute route) noexcept {
  switch (route) {
  case PipelineRoute::NestedSeed:
    return PipelineNestedPhase::Seed;
  case PipelineRoute::NestedAction:
    return PipelineNestedPhase::Action;
  case PipelineRoute::NestedFold:
    return PipelineNestedPhase::Fold;
  case PipelineRoute::Ordinary:
    return PipelineNestedPhase::None;
  }
  return PipelineNestedPhase::None;
}

[[nodiscard]] Location
publication_location(const PipelineBuildState &build,
                     const PipelineBuildPublicationEdge &edge) noexcept {
  Location location{};
  if (edge.step.value >= build.steps.size()) {
    return location;
  }
  const PipelineBuildStep &step = build.steps[edge.step.value];
  location.step = step.logical_step;
  location.iteration = step.iteration;
  location.nested_phase = publication_phase(step.route);
  return location;
}

struct ScheduledPublication final {
  PipelinePublicationPlan plan;
  node::accel::detail::PreparedKernelPublicationIdentity identity;
};

[[nodiscard]] Result<PipelinePhysicalOutputOrdinal>
resolve_publication_output(const PipelineBuildState &build,
                           const PipelineBuildPublicationEdge &edge,
                           const Location location) {
  if (edge.step.value >= build.steps.size()) {
    return Result<PipelinePhysicalOutputOrdinal>::fail(Reason::PipelineInvalid,
                                                       location);
  }
  const PipelineBuildStep &step = build.steps[edge.step.value];
  auto projection = project_outputs(step);
  if (!projection || edge.output.value >= step.outputs.size()) {
    return Result<PipelinePhysicalOutputOrdinal>::fail(
        projection ? Reason::PipelineInvalid : projection.reason(), location);
  }
  const std::uint32_t physical =
      projection->logical_to_physical[edge.output.value];
  if (physical >= projection->physical_count) {
    return Result<PipelinePhysicalOutputOrdinal>::fail(Reason::PipelineInvalid,
                                                       location);
  }
  return Result<PipelinePhysicalOutputOrdinal>::success({.value = physical});
}

[[nodiscard]] Result<ScheduledPublication>
plan_window_publication(const PipelineBuildState &build,
                        const PipelineBuildWindowPublication &publication,
                        const std::span<const std::uint32_t> window_states,
                        PipelineScheduleResources &resources) {
  const PipelineBuildPublicationEdge &edge = publication.edge;
  const Location location = publication_location(build, edge);
  if (edge.step.value >= build.steps.size() ||
      edge.step.value >= window_states.size() ||
      window_states[edge.step.value] == PipelineResourceUnassigned) {
    return Result<ScheduledPublication>::fail(Reason::PipelineInvalid,
                                              location);
  }
  const PipelineBuildStep &first = build.steps[edge.step.value];
  auto physical = resolve_publication_output(build, edge, location);
  if (!physical) {
    return Result<ScheduledPublication>::fail(physical.reason(),
                                              physical.location());
  }
  auto source = resources.admit(edge.source);
  auto target = resources.admit(edge.target);
  auto count = resources.admit(publication.count);
  if (!source || !target || !count) {
    return Result<ScheduledPublication>::fail(!source   ? source.reason()
                                              : !target ? target.reason()
                                                        : count.reason(),
                                              location);
  }
  if (publication.maximum == 0u || publication.tile == 0u ||
      publication.tile > publication.maximum) {
    return Result<ScheduledPublication>::fail(Reason::PipelineInvalid,
                                              location);
  }
  if (publication.maximum > std::numeric_limits<std::uint32_t>::max() ||
      publication.tile > std::numeric_limits<std::uint32_t>::max()) {
    return Result<ScheduledPublication>::fail(Reason::PipelineCapacity,
                                              location);
  }

  node::accel::detail::PreparedKernelPublicationIdentity identity{};
  identity.state = window_states[edge.step.value];
  identity.maximum = static_cast<std::uint32_t>(publication.maximum);
  identity.tile = static_cast<std::uint32_t>(publication.tile);
  identity.kind = static_cast<std::uint8_t>(PipelinePublishKind::Window);
  for (auto &source_identity : identity.sources) {
    if (!publication_view_identity(edge.source, *source,
                                   rund::kernel::kResidentUsageRead,
                                   source_identity)) {
      return Result<ScheduledPublication>::fail(Reason::PipelineCapacity,
                                                location);
    }
  }
  if (!publication_view_identity(edge.target, *target,
                                 rund::kernel::kResidentUsageWrite,
                                 identity.target) ||
      !publication_view_identity(publication.count, *count,
                                 rund::kernel::kResidentUsageRead,
                                 identity.count)) {
    return Result<ScheduledPublication>::fail(Reason::PipelineCapacity,
                                              location);
  }
  if (!PipelineScheduleResources::append(resources.publication_accesses,
                                         edge.source, 0u, *source,
                                         resource::AccessMode::Read) ||
      !PipelineScheduleResources::append(resources.publication_accesses,
                                         edge.target, 0u, *target,
                                         resource::AccessMode::Write) ||
      !PipelineScheduleResources::append(resources.publication_accesses,
                                         publication.count, 0u, *count,
                                         resource::AccessMode::Read)) {
    return Result<ScheduledPublication>::fail(Reason::PipelineCapacity,
                                              location);
  }
  if (first.route != PipelineRoute::NestedFold ||
      build.steps.size() - edge.step.value < 3u) {
    return Result<ScheduledPublication>::fail(Reason::PipelineInvalid,
                                              location);
  }
  for (std::size_t route = 0u; route < 3u; ++route) {
    const std::size_t step_index = edge.step.value + route;
    if (build.steps[step_index].route != PipelineRoute::NestedFold ||
        !PipelineScheduleResources::append(
            resources.accesses, publication.count,
            static_cast<std::uint32_t>(step_index), *count,
            resource::AccessMode::Read) ||
        !PipelineScheduleResources::append(
            resources.accesses, edge.target,
            static_cast<std::uint32_t>(step_index), *target,
            resource::AccessMode::Write)) {
      return Result<ScheduledPublication>::fail(Reason::PipelineInvalid,
                                                location);
    }
  }

  return Result<ScheduledPublication>::success(ScheduledPublication{
      .plan =
          PipelineWindowPublicationPlan{
              .source = *source,
              .count = *count,
              .target = *target,
              .state = identity.state,
              .output = *physical,
              .maximum = identity.maximum,
              .tile = identity.tile,
          },
      .identity = identity,
  });
}

[[nodiscard]] Result<ScheduledPublication>
plan_terminal_publication(const PipelineBuildState &build,
                          const PipelineBuildTerminalPublication &publication,
                          const std::span<const std::uint32_t> window_states,
                          PipelineScheduleResources &resources) {
  const PipelineBuildPublicationEdge &edge = publication.edge;
  const Location location = publication_location(build, edge);
  if (edge.step.value >= build.steps.size() ||
      edge.step.value >= window_states.size() ||
      window_states[edge.step.value] == PipelineResourceUnassigned) {
    return Result<ScheduledPublication>::fail(Reason::PipelineInvalid,
                                              location);
  }
  const PipelineBuildStep &first = build.steps[edge.step.value];
  auto physical = resolve_publication_output(build, edge, location);
  if (!physical) {
    return Result<ScheduledPublication>::fail(physical.reason(),
                                              physical.location());
  }
  if (physical->value >= first.inputs.size()) {
    return Result<ScheduledPublication>::fail(Reason::PipelineInvalid,
                                              location);
  }
  auto source = resources.admit(edge.source);
  auto target = resources.admit(edge.target);
  if (!source || !target) {
    return Result<ScheduledPublication>::fail(
        source ? target.reason() : source.reason(), location);
  }

  const std::uint32_t bound = [&]() -> std::uint32_t {
    if (first.nested == 0u) {
      return first.iteration_bound;
    }
    const std::size_t nested_index = first.nested - 1u;
    return nested_index < build.nested_windows.size()
               ? static_cast<std::uint32_t>(
                     build.nested_windows[nested_index].seed_count)
               : 0u;
  }();
  if (bound == 0u) {
    return Result<ScheduledPublication>::fail(Reason::PipelineInvalid,
                                              location);
  }

  node::accel::detail::PreparedKernelPublicationIdentity identity{};
  identity.state = window_states[edge.step.value];
  identity.kind = static_cast<std::uint8_t>(PipelinePublishKind::Terminal);
  identity.final = 1u + ((bound - 1u) & 1u);
  if (!publication_view_identity(edge.target, *target,
                                 rund::kernel::kResidentUsageWrite,
                                 identity.target)) {
    return Result<ScheduledPublication>::fail(Reason::PipelineCapacity,
                                              location);
  }

  PipelineTerminalPublicationPlan planned{};
  planned.sources.fill(PipelineResourceUnassigned);
  planned.target = *target;
  planned.state = identity.state;
  planned.output = *physical;
  planned.final = identity.final;
  for (std::size_t bank = 0u; bank < planned.sources.size(); ++bank) {
    const PipelineBinding *binding = nullptr;
    if (bank == 0u) {
      binding = &first.inputs[physical->value];
    } else {
      std::size_t step_index = edge.step.value;
      if (bank == 2u && step_index + 1u < build.steps.size() &&
          window_states[step_index + 1u] == window_states[edge.step.value]) {
        ++step_index;
      }
      auto bank_projection = project_outputs(build.steps[step_index]);
      if (!bank_projection ||
          physical->value >= bank_projection->physical_count) {
        return Result<ScheduledPublication>::fail(
            bank_projection ? Reason::PipelineInvalid
                            : bank_projection.reason(),
            location);
      }
      const std::uint32_t authored =
          bank_projection->physical_sources[physical->value];
      if (authored >= build.steps[step_index].outputs.size()) {
        return Result<ScheduledPublication>::fail(Reason::PipelineInvalid,
                                                  location);
      }
      binding = &build.steps[step_index].outputs[authored];
    }
    auto ordinal = resources.admit(*binding);
    if (!ordinal || !publication_view_identity(*binding, *ordinal,
                                               rund::kernel::kResidentUsageRead,
                                               identity.sources[bank])) {
      return Result<ScheduledPublication>::fail(
          ordinal ? Reason::PipelineCapacity : ordinal.reason(), location);
    }
    planned.sources[bank] = *ordinal;
  }
  if (identity.final >= planned.sources.size() ||
      planned.sources[identity.final] != *source ||
      !PipelineScheduleResources::append(resources.publication_accesses,
                                         edge.source, 0u, *source,
                                         resource::AccessMode::Read) ||
      !PipelineScheduleResources::append(resources.publication_accesses,
                                         edge.target, 0u, *target,
                                         resource::AccessMode::Write)) {
    return Result<ScheduledPublication>::fail(Reason::PipelineInvalid,
                                              location);
  }

  return Result<ScheduledPublication>::success(ScheduledPublication{
      .plan = planned,
      .identity = identity,
  });
}

} // namespace

Result<PipelineScheduleSuccess>
plan_pipeline_publications(const PipelineBuildState &build,
                           const std::span<const std::uint32_t> window_states,
                           PipelineScheduleResources &resources,
                           PipelineMemoryPlan &plan) {
  plan.publications.clear();
  plan.publications.reserve(build.publications.size());
  node::accel::detail::SeedPreparedKernelPublicationFingerprint(
      plan.publication_fingerprint_hi, plan.publication_fingerprint_lo);
  for (const PipelineBuildPublication &publication : build.publications) {
    const auto *window =
        std::get_if<PipelineBuildWindowPublication>(&publication);
    auto scheduled =
        window != nullptr
            ? plan_window_publication(build, *window, window_states, resources)
            : plan_terminal_publication(
                  build,
                  std::get<PipelineBuildTerminalPublication>(publication),
                  window_states, resources);
    if (!scheduled) {
      return Result<PipelineScheduleSuccess>::fail(scheduled.reason(),
                                                   scheduled.location());
    }
    node::accel::detail::MixPreparedKernelPublicationFingerprint(
        plan.publication_fingerprint_hi, plan.publication_fingerprint_lo,
        scheduled->identity);
    plan.publications.push_back(std::move(scheduled->plan));
  }
  return Result<PipelineScheduleSuccess>::success({});
}

} // namespace rund::compute::detail
