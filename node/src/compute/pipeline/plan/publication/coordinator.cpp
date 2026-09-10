#include "../../state/assembly.hpp"
#include "internal.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <variant>

namespace rund::compute::detail {

Result<PipelineScheduleSuccess>
plan_pipeline_publications(const PipelineBuildState &build,
                           const std::span<const std::uint32_t> window_states,
                           PipelineScheduleResources &resources,
                           PipelineMemoryPlan &plan) {
  auto controls = pipeline_publication_detail::plan_window_controls(
      build, window_states, resources, plan);
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
            ? pipeline_publication_detail::plan_window_publication(
                  build, *window, window_states, plan.step_resources,
                  plan.window_controls, resources)
            : pipeline_publication_detail::plan_terminal_publication(
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
