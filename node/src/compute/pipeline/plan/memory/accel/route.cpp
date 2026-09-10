#include "../../../state/assembly.hpp"
#include "local.hpp"

#include "../../arena.hpp"
#include "../../compare.hpp"
#include "../../prepare.hpp"
#include "../../resource.hpp"

#include "../../../../../accel/kernel/recurrence.hpp"
#include "../../../../buffer/local.hpp"
#include "../../../../cpu/run/state.hpp"
#include "../../../../job/local.hpp"
#include "../../../../memory/arena.hpp"
#include "../../../../status.hpp"
#include "../../../../type.hpp"

#include <kernel/core/checked.hpp>
#include <rund/compute/abi/observe.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace rund::compute::detail {
namespace {

[[nodiscard]] Status append_program_binding_identity(
    const PipelineResolvedViewPlan &view, const std::uint32_t usage,
    std::vector<node::accel::detail::PreparedKernelProgramBindingIdentity>
        &out) {
  if (view.element_bytes == 0u || view.count == 0u || view.stride_bytes == 0u) {
    return Status::fail(Reason::PipelineCapacity);
  }
  out.push_back(node::accel::detail::PreparedKernelProgramBindingIdentity{
      .offset_bytes = view.offset_bytes,
      .element_bytes = view.element_bytes,
      .stride_bytes = view.stride_bytes,
      .count = view.count,
      .usage = usage,
  });
  return Status::success();
}

[[nodiscard]] Status plan_program_binding_identities(
    const PipelineBuildStep &step, const std::size_t step_index,
    const PipelineMemoryPlan &plan,
    std::vector<node::accel::detail::PreparedKernelProgramBindingIdentity>
        &out) {
  const ProgramState *const program = step.program.get();
  if (program == nullptr || step_index >= plan.workspace_routes.size() ||
      step_index >= plan.step_resources.size() ||
      plan.steps.size() != plan.workspace_routes.size() + 1u) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const PipelineStepResourcePlan &sealed = plan.step_resources[step_index];
  out.clear();
  out.reserve(program->graph_bindings.size());
  for (const GraphRunBinding graph_binding : program->graph_bindings) {
    if (graph_binding.value_index >= program->graph_value_routes.size()) {
      return Status::fail(Reason::GraphBindingInvalid);
    }
    const std::uint32_t usage =
        graph_binding.role == kernel::BufferRole::Read
            ? kernel::kResidentUsageRead
        : graph_binding.role == kernel::BufferRole::Write
            ? kernel::kResidentUsageWrite
            : 0u;
    if (usage == 0u) {
      return Status::fail(Reason::GraphBindingInvalid);
    }
    const GraphValueRoute route =
        program->graph_value_routes[graph_binding.value_index];
    const PipelineResolvedViewPlan *view = nullptr;
    if (route.source == GraphBindSource::Input) {
      if (route.index >= sealed.inputs.size()) {
        return Status::fail(Reason::GraphBindingInvalid);
      }
      view = &sealed.inputs[route.index];
    } else if (route.source == GraphBindSource::Output) {
      if (route.index >= sealed.physical_sources.size()) {
        return Status::fail(Reason::GraphBindingInvalid);
      }
      const std::uint32_t source = sealed.physical_sources[route.index];
      if (source >= sealed.outputs.size()) {
        return Status::fail(Reason::GraphBindingInvalid);
      }
      view = &sealed.outputs[source].view;
    }
    if (view != nullptr) {
      const Status appended =
          append_program_binding_identity(*view, usage, out);
      if (!appended) {
        return appended;
      }
      continue;
    }
    if (route.source != GraphBindSource::Internal ||
        route.index >= program->chunks.size() || route.element_bytes == 0u ||
        route.count == 0u) {
      return Status::fail(Reason::GraphBindingInvalid);
    }
    const PipelineWorkspaceRoute workspace = plan.workspace_routes[step_index];
    if (!workspace.present()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::size_t workspace_index = workspace.owner;
    if (workspace_index >= step_index + 1u ||
        workspace_index + 1u >= plan.steps.size() ||
        workspace_index >= plan.workspace_routes.size() ||
        !plan.workspace_routes[workspace_index].owns(workspace_index)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::size_t begin = plan.steps[workspace_index];
    const std::size_t end = plan.steps[workspace_index + 1u];
    if (begin > end || end > plan.offsets.size() ||
        end - begin != program->chunk_order.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    std::size_t rank = 0u;
    while (rank < program->chunk_order.size() &&
           program->chunk_order[rank] != route.index) {
      ++rank;
    }
    std::uint64_t workspace_offset = 0u;
    std::uint64_t offset_bytes = 0u;
    if (rank == program->chunk_order.size() ||
        !kernel::checked::mul(
            static_cast<std::uint64_t>(plan.offsets[begin + rank]),
            static_cast<std::uint64_t>(memory::Word), workspace_offset) ||
        !kernel::checked::add(workspace_offset, route.offset_bytes,
                              offset_bytes)) {
      return Status::fail(Reason::PipelineCapacity);
    }
    out.push_back(node::accel::detail::PreparedKernelProgramBindingIdentity{
        .offset_bytes = offset_bytes,
        .element_bytes = route.element_bytes,
        .stride_bytes = route.element_bytes,
        .count = route.count,
        .usage = usage,
    });
  }
  return out.size() == program->graph_bindings.size()
             ? Status::success()
             : Status::fail(Reason::GraphBindingInvalid);
}
} // namespace

Status collect_pipeline_accel_routes(
    const PipelineBuildState &build, PipelineMemoryPlan &plan,
    PipelineAccelPreparationDraft &draft) {
  const auto &entry_counts = draft.entry_counts;
  const auto &occurrence_counts = draft.occurrence_counts;
  const auto &window_counts = draft.window_counts;
  const auto &nested_group_counts = draft.nested_group_counts;
  const auto &map_recurrence_group_counts =
      draft.map_recurrence_group_counts;
  const auto &map_recurrence_history_group_counts =
      draft.map_recurrence_history_group_counts;
  const auto &recurrence_hi = draft.recurrence_hi;
  const auto &recurrence_lo = draft.recurrence_lo;

  auto &routes = draft.routes;
  auto &route_program_bindings = draft.route_program_bindings;
  routes.clear();
  route_program_bindings.clear();
  routes.reserve(build.steps.size());
  route_program_bindings.reserve(build.steps.size());
  draft.route_copies = build.state_pairs.empty() ? 1u : 2u;
  const std::uint32_t route_copies = draft.route_copies;
  const bool has_arena = !plan.view_chunks.empty() || !plan.scratch.empty();
    for (std::size_t index = 0u; index < build.steps.size(); ++index) {
      const PipelineBuildStep &step = build.steps[index];
      if (plan.job_owners[index] != index || step.program == nullptr ||
          step.program->empty()) {
        continue;
      }
      const PipelineWorkspaceRoute workspace = plan.workspace_routes[index];
      if ((has_arena && !workspace.present()) ||
          (workspace.present() &&
           (workspace.owner > index || workspace.owner >= build.steps.size() ||
            !plan.workspace_routes[workspace.owner].owns(workspace.owner) ||
            build.steps[workspace.owner].program != step.program))) {
        return Status::fail(Reason::PipelineInvalid);
      }
      if (step.program->accel == nullptr || entry_counts[index] == 0u) {
        return Status::fail(Reason::AccelProgramInvalid);
      }
      route_program_bindings.emplace_back();
      const Status binding_identity = plan_program_binding_identities(
          step, index, plan, route_program_bindings.back());
      if (!binding_identity) {
        return binding_identity;
      }
      routes.push_back(node::accel::detail::PreparedKernelProgramRoute{
          .kernel = &step.program->accel->kernel,
          .tile_count = step.program->count,
          .views = has_arena ? &plan.views[index] : nullptr,
          .scratch = has_arena ? &plan.scratch : nullptr,
          .program_bindings = route_program_bindings.back(),
          .entry_count = entry_counts[index],
          .occurrence_count = occurrence_counts[index],
          .window_count = window_counts[index],
          .nested_group_count = nested_group_counts[index],
          .map_recurrence_group_count = map_recurrence_group_counts[index],
          .map_recurrence_history_group_count =
              map_recurrence_history_group_counts[index],
          .recurrence_fingerprint_hi = recurrence_hi[index],
          .recurrence_fingerprint_lo = recurrence_lo[index],
          .route_copies = route_copies,
      });
    }
    if (routes.empty()) {
      // A Pipeline may still contain declared logical steps when every
      // Program is canonical zero work.  No accelerator route, template, or
      // native command exists in that case, but bind() still consumes the
      // frozen planning decision.  Preserve that distinction as an admitted
      // zero reservation instead of leaving the default (invalid) sentinel.
      plan.accel_preparation.ok = true;
      plan.accel_preparation.reason = "ok";
      return Status::success();
    }

  return Status::success();
}

} // namespace rund::compute::detail
