#include "publication.hpp"

#include "../output.hpp"
#include "compare.hpp"
#include "contract.hpp"

#include "../../size.hpp"
#include "../../type.hpp"

#include <kernel/program/compute/binding/model.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace rund::compute::detail {

namespace {

[[nodiscard]] Location
publication_location(const PipelineBuildState &build,
                     const PipelinePublicationStepOrdinal step) noexcept {
  Location location{};
  if (step.value >= build.steps.size()) {
    return location;
  }
  const PipelineBuildStep &authored = build.steps[step.value];
  location.step = authored.logical_step;
  location.iteration = authored.iteration;
  location.nested_phase = pipeline_nested_phase(authored.route);
  return location;
}

[[nodiscard]] Result<PipelinePublicationViewPlan>
seal_publication_view(PipelineScheduleResources &resources,
                      const PipelineBinding &binding,
                      const std::optional<ResourceAccess> expected_access,
                      const std::uint32_t usage, const Location location) {
  return resources.publication_view(binding, binding.type, binding.count,
                                    binding.format, expected_access, usage,
                                    location);
}

struct SealedPublicationOutput final {
  const PipelineResolvedOutputPlan *output{};
  PipelinePhysicalOutputOrdinal physical{};
};

[[nodiscard]] Result<SealedPublicationOutput> resolve_publication_output(
    const PipelineBuildState &build,
    const std::span<const PipelineStepResourcePlan> steps,
    const PipelineBuildOutputCoordinate coordinate, const Location location) {
  auto projected = resolve_build_output(build, coordinate);
  if (!projected || coordinate.step.value >= steps.size()) {
    return Result<SealedPublicationOutput>::fail(
        projected ? Reason::PipelineInvalid : projected.reason(), location);
  }
  const PipelineStepResourcePlan &step = steps[coordinate.step.value];
  if (projected->physical.value >= step.physical_sources.size()) {
    return Result<SealedPublicationOutput>::fail(Reason::PipelineInvalid,
                                                 location);
  }
  const std::uint32_t source = step.physical_sources[projected->physical.value];
  if (source >= step.outputs.size() ||
      step.outputs[source].physical != projected->physical.value) {
    return Result<SealedPublicationOutput>::fail(Reason::PipelineInvalid,
                                                 location);
  }
  return Result<SealedPublicationOutput>::success({
      .output = &step.outputs[source],
      .physical = projected->physical,
  });
}

[[nodiscard]] Result<PipelineScheduleSuccess>
plan_window_controls(const PipelineBuildState &build,
                     const std::span<const std::uint32_t> window_states,
                     PipelineScheduleResources &resources,
                     PipelineMemoryPlan &plan) {
  if (window_states.size() != build.steps.size()) {
    return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
  }
  plan.window_controls.clear();
  plan.window_controls.reserve(build.window_controls.size());
  for (std::size_t state = 0u; state < build.window_controls.size(); ++state) {
    const PipelineBuildWindowControl &authored = build.window_controls[state];
    const PipelineBuildWindowControlOrdinal control_ordinal{
        .value = static_cast<std::uint32_t>(state),
    };
    auto anchors = resolve_build_window_anchors(build, control_ordinal);
    if (!anchors || anchors->count_step.value >= plan.step_resources.size() ||
        anchors->count_step.value >= window_states.size() ||
        anchors->terminal_step.value >= build.steps.size() ||
        anchors->terminal_step.value >= window_states.size() ||
        window_states[anchors->count_step.value] != state ||
        window_states[anchors->terminal_step.value] != state ||
        authored.count_input >=
            plan.step_resources[anchors->count_step.value].inputs.size() ||
        anchors->count_step.value > std::numeric_limits<std::uint32_t>::max() ||
        authored.count_input > std::numeric_limits<std::uint32_t>::max() ||
        authored.maximum == 0u || authored.tile == 0u ||
        authored.tile > authored.maximum ||
        authored.maximum > std::numeric_limits<std::uint32_t>::max() ||
        authored.tile > std::numeric_limits<std::uint32_t>::max()) {
      return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
    }
    const PipelineBuildStep &count_step =
        build.steps[anchors->count_step.value];
    const PipelineBuildStep &terminal_step =
        build.steps[anchors->terminal_step.value];
    if (authored.nested != 0u) {
      const std::size_t nested_index = authored.nested - 1u;
      if (nested_index >= build.nested_windows.size()) {
        return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
      }
      if (count_step.nested != authored.nested ||
          terminal_step.nested != authored.nested) {
        return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
      }
    } else if (count_step.iteration != 0u || count_step.nested != 0u ||
               terminal_step.nested != 0u) {
      return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
    }
    auto selected = resolve_build_window_final(build, control_ordinal);
    if (!selected) {
      return Result<PipelineScheduleSuccess>::fail(selected.reason());
    }
    const PipelineResolvedViewPlan &resolved_count =
        plan.step_resources[anchors->count_step.value]
            .inputs[authored.count_input];
    auto count = resources.publication_view(
        resolved_count, rund::kernel::kResidentUsageRead, {});
    if (!count) {
      return Result<PipelineScheduleSuccess>::fail(count.reason(),
                                                   count.location());
    }
    if (count->type != Type::U32 || count->identity.count != 1u ||
        count->identity.element_bytes != sizeof(std::uint32_t) ||
        count->identity.stride_bytes != sizeof(std::uint32_t)) {
      return Result<PipelineScheduleSuccess>::fail(Reason::BindingInvalid);
    }
    std::uint32_t terminal_output = std::numeric_limits<std::uint32_t>::max();
    if (authored.terminal != NoWindowTerminal) {
      if (anchors->terminal_step.value >= plan.step_resources.size() ||
          authored.terminal >= plan.step_resources[anchors->terminal_step.value]
                                   .outputs.size()) {
        return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
      }
      const PipelineStepResourcePlan &sealed_terminal =
          plan.step_resources[anchors->terminal_step.value];
      terminal_output = sealed_terminal.outputs[authored.terminal].physical;
      if (terminal_output >= sealed_terminal.physical_sources.size()) {
        return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
      }
    }
    plan.window_controls.push_back(PipelineWindowControl{
        .count = *count,
        .count_input = static_cast<std::uint32_t>(authored.count_input),
        .maximum = static_cast<std::uint32_t>(authored.maximum),
        .tile = static_cast<std::uint32_t>(authored.tile),
        .terminal = authored.terminal == NoWindowTerminal
                        ? std::numeric_limits<std::uint32_t>::max()
                        : static_cast<std::uint32_t>(authored.terminal),
        .terminal_output = terminal_output,
        .expected = authored.expected,
        .final = selected->bank,
    });
    const PipelineWindowControl &control = plan.window_controls.back();
    if (authored.nested != 0u) {
      const PipelineBuildNestedWindow &nested =
          build.nested_windows[authored.nested - 1u];
      if (!nested.shape.valid()) {
        return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
      }
      for (std::size_t route = 0u; route < nested.shape.fold_count(); ++route) {
        const std::size_t fold = nested.shape.fold_first() + route;
        if (fold >= build.steps.size() ||
            build.steps[fold].route != PipelineRoute::NestedFold ||
            !PipelineScheduleResources::append(
                resources.accesses, control.count.identity,
                static_cast<std::uint32_t>(fold))) {
          return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
        }
      }
    }
  }
  return Result<PipelineScheduleSuccess>::success({});
}

[[nodiscard]] Result<PipelinePublicationPlan> plan_window_publication(
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

  // All retained Fold routes must name the same append-only Tile producer.
  // Otherwise route choice would become a hidden second Window source law.
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

[[nodiscard]] Result<PipelinePublicationPlan> plan_terminal_publication(
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

[[nodiscard]] node::accel::detail::PreparedKernelPublicationViewIdentity
project_view(const PipelinePublicationViewIdentity &view) noexcept {
  return node::accel::detail::PreparedKernelPublicationViewIdentity{
      .backing_bytes = view.backing_bytes,
      .offset_bytes = view.offset_bytes,
      .count = view.count,
      .stride_bytes = view.stride_bytes,
      .element_bytes = view.element_bytes,
      .resource_ordinal = view.resource_ordinal,
      .usage = view.usage,
  };
}

} // namespace

bool mix_pipeline_publication_public_identity(
    PipelineHash &hash, const PipelinePublicationPlan &publication,
    const PipelineWindowControl &control) noexcept {
  static_assert(static_cast<std::uint8_t>(PipelinePublicationKind::Terminal) ==
                0u);
  static_assert(static_cast<std::uint8_t>(PipelinePublicationKind::Window) ==
                1u);

  const auto *window = std::get_if<PipelineWindowPublicationPlan>(&publication);
  const auto *terminal =
      std::get_if<PipelineTerminalPublicationPlan>(&publication);
  if ((window == nullptr && terminal == nullptr) ||
      control.final < PipelineWindow::first ||
      control.final > PipelineWindow::second) {
    return false;
  }
  const PipelinePublicationViewPlan &source =
      window != nullptr ? window->source : terminal->sources[control.final];
  const PipelinePublicationViewPlan &target =
      window != nullptr ? window->target.view : terminal->target.view;
  const PipelinePublicationViewIdentity &source_identity = source.identity;
  const PipelinePublicationViewIdentity &target_identity = target.identity;
  const PipelinePublicationViewPlan *const count =
      window != nullptr ? &control.count : nullptr;
  if (!valid_type(source.type) || !valid_format(source.type, source.format) ||
      source_identity.element_bytes == 0u ||
      target_identity.offset_bytes % source_identity.element_bytes != 0u ||
      target_identity.stride_bytes % source_identity.element_bytes != 0u ||
      (count != nullptr &&
       (count->identity.element_bytes == 0u ||
        count->identity.offset_bytes % count->identity.element_bytes != 0u))) {
    return false;
  }

  hash.number(source_identity.resource_ordinal);
  hash.number(target_identity.resource_ordinal);
  hash.number(static_cast<std::uint64_t>(source.type));
  hash.format(source.format);
  hash.number(source_identity.count);
  hash.number(target_identity.offset_bytes / source_identity.element_bytes);
  hash.number(target_identity.stride_bytes / source_identity.element_bytes);
  hash.number(source_identity.element_bytes);
  hash.number(static_cast<std::uint64_t>(window != nullptr ? window->state
                                                           : terminal->state) +
              1u);
  hash.number(window != nullptr ? window->output.value
                                : terminal->output.value);
  hash.byte(static_cast<std::uint8_t>(pipeline_publication_kind(publication)));
  hash.number(count == nullptr ? 0u
                               : count->identity.offset_bytes /
                                     count->identity.element_bytes);
  hash.number(window != nullptr ? control.maximum : 0u);
  hash.number(window != nullptr ? control.tile : 0u);
  return true;
}

node::accel::detail::PreparedKernelPublicationIdentity
project_pipeline_publication_identity(
    const PipelinePublicationPlan &publication,
    const PipelineWindowControl &control,
    const std::uint32_t outer_bound) noexcept {
  node::accel::detail::PreparedKernelPublicationIdentity identity{};
  if (const auto *terminal =
          std::get_if<PipelineTerminalPublicationPlan>(&publication)) {
    for (std::size_t bank = 0u; bank < terminal->sources.size(); ++bank) {
      identity.sources[bank] = project_view(terminal->sources[bank].identity);
    }
    identity.target = project_view(terminal->target.view.identity);
    identity.state = terminal->state;
    identity.final = control.final;
    identity.kind =
        node::accel::detail::PreparedKernelPublicationKind::Terminal;
    return identity;
  }
  const PipelineWindowPublicationPlan &window =
      std::get<PipelineWindowPublicationPlan>(publication);
  const auto source = project_view(window.source.identity);
  std::fill(std::begin(identity.sources), std::end(identity.sources), source);
  identity.count = project_view(control.count.identity);
  identity.target = project_view(window.target.view.identity);
  identity.state = window.state;
  identity.maximum = control.maximum;
  identity.tile = control.tile;
  identity.outer_bound = outer_bound;
  identity.kind = node::accel::detail::PreparedKernelPublicationKind::Window;
  return identity;
}

Result<PipelineScheduleSuccess>
plan_pipeline_publications(const PipelineBuildState &build,
                           const std::span<const std::uint32_t> window_states,
                           PipelineScheduleResources &resources,
                           PipelineMemoryPlan &plan) {
  auto controls = plan_window_controls(build, window_states, resources, plan);
  if (!controls) {
    return controls;
  }
  plan.publications.clear();
  plan.publications.reserve(build.publications.size());
  node::accel::detail::SeedPreparedKernelPublicationFingerprint(
      plan.publication_fingerprint_hi, plan.publication_fingerprint_lo);
  for (const PipelineBuildPublication &publication : build.publications) {
    const auto *window =
        std::get_if<PipelineBuildWindowPublication>(&publication);
    auto planned =
        window != nullptr
            ? plan_window_publication(build, *window, window_states,
                                      plan.step_resources, plan.window_controls,
                                      resources)
            : plan_terminal_publication(
                  build,
                  std::get<PipelineBuildTerminalPublication>(publication),
                  window_states, plan.step_resources, plan.window_controls,
                  resources);
    if (!planned) {
      return Result<PipelineScheduleSuccess>::fail(planned.reason(),
                                                   planned.location());
    }
    const std::uint32_t state =
        std::visit([](const auto &typed) { return typed.state; }, *planned);
    if (state >= plan.window_controls.size()) {
      return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
    }
    const node::accel::detail::NestedTemplateShape *const nested_shape =
        window != nullptr ? pipeline_build_nested_shape(build, state) : nullptr;
    if (window != nullptr &&
        (nested_shape == nullptr || !nested_shape->valid())) {
      return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
    }
    const auto identity = project_pipeline_publication_identity(
        *planned, plan.window_controls[state],
        nested_shape == nullptr ? 0u : nested_shape->outer_bound());
    node::accel::detail::MixPreparedKernelPublicationFingerprint(
        plan.publication_fingerprint_hi, plan.publication_fingerprint_lo,
        identity);
    plan.publications.push_back(std::move(*planned));
  }
  for (PipelineWindowControl &control : plan.window_controls) {
    if (control.terminal_output == std::numeric_limits<std::uint32_t>::max()) {
      continue;
    }
    for (std::size_t index = 0u; index < plan.publications.size(); ++index) {
      const auto *terminal = std::get_if<PipelineTerminalPublicationPlan>(
          &plan.publications[index]);
      if (terminal == nullptr ||
          terminal->state >= plan.window_controls.size() ||
          &control != &plan.window_controls[terminal->state] ||
          terminal->output.value != control.terminal_output) {
        continue;
      }
      if (control.terminal_publication !=
          std::numeric_limits<std::uint32_t>::max()) {
        return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
      }
      control.terminal_publication = static_cast<std::uint32_t>(index);
    }
    if (control.terminal_publication ==
        std::numeric_limits<std::uint32_t>::max()) {
      return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
    }
  }
  return Result<PipelineScheduleSuccess>::success({});
}

} // namespace rund::compute::detail
