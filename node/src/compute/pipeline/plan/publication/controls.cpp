#include "../../state/assembly.hpp"
#include "internal.hpp"

#include <kernel/program/compute/binding/model.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::compute::detail::pipeline_publication_detail {

Result<PipelineScheduleSuccess>
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

} // namespace rund::compute::detail::pipeline_publication_detail
