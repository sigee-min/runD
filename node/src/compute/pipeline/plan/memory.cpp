#include "arena.hpp"
#include "compare.hpp"
#include "local.hpp"
#include "prepare.hpp"
#include "resource.hpp"

#include "../../../accel/kernel/recurrence.hpp"
#include "../../backend.hpp"
#include "../../buffer/local.hpp"
#include "../../cpu/run/state.hpp"
#include "../../job/local.hpp"
#include "../../memory/arena.hpp"
#include "../../status.hpp"
#include "../../type.hpp"

#include <kernel/core/checked.hpp>
#include <rund/compute/abi/observe.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <optional>
#include <span>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rund::compute::detail {
namespace {

[[nodiscard]] bool add_extent(std::uint64_t &total, const std::size_t count,
                              const std::uint64_t width) noexcept {
  if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
    if (count > std::numeric_limits<std::uint64_t>::max()) {
      return false;
    }
  }
  std::uint64_t bytes = 0u;
  return kernel::checked::mul(static_cast<std::uint64_t>(count), width,
                              bytes) &&
         kernel::checked::add(total, bytes, total);
}

[[nodiscard]] Result<std::size_t>
pipeline_output_count(const PipelineMemoryPlan &plan) {
  std::size_t count = 0u;
  for (const PipelineResolvedResourcePlan &resource : plan.resources) {
    if (!resource.output) {
      continue;
    }
    if (count == std::numeric_limits<std::size_t>::max()) {
      return Result<std::size_t>::fail(Reason::PipelineCapacity);
    }
    ++count;
  }
  return Result<std::size_t>::success(count);
}

[[nodiscard]] Status
seal_pipeline_workspace_routes(const PipelineBuildState &build,
                               PipelineMemoryPlan &plan) {
  if (plan.views.size() != build.steps.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  plan.workspace_routes.assign(build.steps.size(), {});
  const bool has_job_arena = !plan.view_chunks.empty() || !plan.scratch.empty();
  for (std::size_t index = 0u; index < build.steps.size(); ++index) {
    const PipelineBuildStep &step = build.steps[index];
    if (step.program == nullptr) {
      return Status::fail(Reason::PipelineInvalid);
    }
    // This is the sole workspace-presence decision. CPU dense-View transfer
    // layouts are deliberately absent: they are owned by each private Job.
    const bool present = !step.program->chunks.empty() ||
                         !plan.views[index].empty() ||
                         (has_job_arena && !step.program->empty());
    if (step.iteration_bound > 1u && step.iteration != 0u) {
      if (index == 0u) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const PipelineBuildStep &previous = build.steps[index - 1u];
      const PipelineWorkspaceRoute previous_route =
          plan.workspace_routes[index - 1u];
      if (previous.program != step.program ||
          previous.logical_step != step.logical_step ||
          previous.iteration_bound != step.iteration_bound ||
          previous.nested != step.nested || previous.route != step.route ||
          previous.iteration == std::numeric_limits<std::uint32_t>::max() ||
          step.iteration != previous.iteration + 1u ||
          previous_route.present() != present) {
        return Status::fail(Reason::PipelineInvalid);
      }
      plan.workspace_routes[index] = previous_route;
      continue;
    }
    if (present) {
      plan.workspace_routes[index].owner = index;
    }
  }
  return Status::success();
}

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

[[nodiscard]] std::size_t
cpu_view_binding_ordinal(const ProgramState &program,
                         const GraphBindSource source,
                         const std::uint32_t port) noexcept {
  for (std::size_t ordinal = 0u; ordinal < program.graph_bindings.size();
       ++ordinal) {
    const std::uint32_t value = program.graph_bindings[ordinal].value_index;
    if (value >= program.graph_value_routes.size()) {
      return std::numeric_limits<std::size_t>::max();
    }
    const GraphValueRoute route = program.graph_value_routes[value];
    if (route.source == source && route.index == port) {
      return ordinal;
    }
  }
  return std::numeric_limits<std::size_t>::max();
}

[[nodiscard]] Status record_cpu_view(PipelinePlan &summary,
                                     const PipelineBuildStep &step,
                                     const PipelineResolvedViewPlan &view,
                                     const std::uint64_t bytes,
                                     const std::size_t ordinal) noexcept {
  if (ordinal == std::numeric_limits<std::size_t>::max() || view.count <= 1u ||
      view.stride == 1u || view.span_bytes == 0u) {
    return Status::fail(Reason::GraphBindingInvalid);
  }
  const auto location = std::tuple{step.logical_step, step.iteration, ordinal};
  const auto largest = std::tuple{summary.view_step, summary.view_iteration,
                                  summary.view_binding};
  if (bytes < summary.view_bytes ||
      (bytes == summary.view_bytes && !(location < largest))) {
    return Status::success();
  }
  summary.view_bytes = bytes;
  summary.view_span_bytes = view.span_bytes;
  summary.view_backing_bytes = view.declared_backing_bytes;
  summary.view_offset_bytes = view.offset_bytes;
  summary.view_stride_bytes = view.stride_bytes;
  summary.view_element_bytes = view.element_bytes;
  summary.view_count = view.count;
  summary.view_alignment = view.alignment;
  summary.view_step = step.logical_step;
  summary.view_iteration = step.iteration;
  summary.view_outer_window = step.route == PipelineRoute::NestedSeed
                                  ? static_cast<std::size_t>(step.iteration)
                                  : std::numeric_limits<std::size_t>::max();
  summary.view_inner_iteration = step.route == PipelineRoute::NestedAction
                                     ? static_cast<std::size_t>(step.iteration)
                                     : std::numeric_limits<std::size_t>::max();
  summary.view_nested_phase = pipeline_nested_phase(step.route);
  summary.view_binding = ordinal;
  return Status::success();
}

[[nodiscard]] Status plan_pipeline_cpu_views(const PipelineBuildState &build,
                                             PipelineMemoryPlan &plan) {
  plan.cpu_view_layouts.clear();
  plan.cpu_view_layouts.resize(build.steps.size());
  if (build.device->backend != Backend::Cpu) {
    return Status::success();
  }
  std::vector<std::optional<CpuViewTransferRequirements>> requirements(
      plan.cpu_programs.size());
  std::vector<JobBufferView> inputs;
  std::vector<JobBufferView> outputs;
  const std::uint64_t copies = build.state_pairs.empty() ? 1u : 2u;
  for (std::size_t index = 0u; index < build.steps.size(); ++index) {
    if (plan.job_owners[index] != index) {
      continue;
    }
    const PipelineBuildStep &step = build.steps[index];
    if (index >= plan.step_resources.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const PipelineStepResourcePlan &sealed = plan.step_resources[index];
    inputs.clear();
    outputs.clear();
    inputs.reserve(sealed.inputs.size());
    outputs.reserve(sealed.physical_sources.size());
    for (const PipelineResolvedViewPlan &view : sealed.inputs) {
      inputs.push_back(PipelineScheduleResources::job_view(view));
    }
    for (const std::uint32_t source : sealed.physical_sources) {
      if (source >= sealed.outputs.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      outputs.push_back(
          PipelineScheduleResources::job_view(sealed.outputs[source].view));
    }
    const CpuViewTransferRequirements *program_requirements = nullptr;
    if (!step.program->empty() && step.program->cpu_graph != nullptr) {
      if (index >= plan.cpu_storage_by_step.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const std::size_t program_index = plan.cpu_storage_by_step[index];
      if (program_index >= requirements.size() ||
          program_index >= plan.cpu_programs.size() ||
          plan.cpu_programs[program_index] != step.program) {
        return Status::fail(Reason::PipelineInvalid);
      }
      std::optional<CpuViewTransferRequirements> &cached =
          requirements[program_index];
      if (!cached.has_value()) {
        auto planned = plan_cpu_view_transfer_requirements(step.program);
        if (!planned) {
          return Status::fail(planned.reason());
        }
        cached.emplace(std::move(planned).value());
      }
      program_requirements = &*cached;
    }
    auto layout = plan_cpu_view_transfers(step.program, inputs, outputs,
                                          program_requirements);
    if (!layout) {
      return Status::fail(layout.reason());
    }
    for (const CpuViewTransferSlot slot : layout->inputs) {
      if (slot.index >= sealed.inputs.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const Status recorded = record_cpu_view(
          plan.summary, step, sealed.inputs[slot.index], slot.bytes,
          cpu_view_binding_ordinal(*step.program, GraphBindSource::Input,
                                   slot.index));
      if (!recorded) {
        return recorded;
      }
    }
    for (const CpuViewTransferSlot slot : layout->outputs) {
      if (slot.index >= sealed.physical_sources.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const std::uint32_t source = sealed.physical_sources[slot.index];
      if (source >= sealed.outputs.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const Status recorded = record_cpu_view(
          plan.summary, step, sealed.outputs[source].view, slot.bytes,
          cpu_view_binding_ordinal(*step.program, GraphBindSource::Output,
                                   slot.index));
      if (!recorded) {
        return recorded;
      }
    }
    const std::size_t input_count = layout->inputs.size();
    const std::size_t output_count = layout->outputs.size();
    if (input_count > std::numeric_limits<std::size_t>::max() - output_count) {
      return Status::fail(Reason::PipelineCapacity);
    }
    const std::size_t transfer_count = input_count + output_count;
    std::uint64_t retained_bytes = 0u;
    std::uint64_t allocations = 0u;
    if (!kernel::checked::mul(layout->bytes, copies, retained_bytes) ||
        !kernel::checked::mul(static_cast<std::uint64_t>(transfer_count),
                              copies, allocations) ||
        !kernel::checked::add(plan.summary.prepared_buffer_bytes,
                              retained_bytes,
                              plan.summary.prepared_buffer_bytes) ||
        !kernel::checked::add(plan.summary.allocation_count, allocations,
                              plan.summary.allocation_count)) {
      return Status::fail(Reason::PipelineCapacity);
    }
    plan.cpu_view_layouts[index] = std::move(layout).value();
  }
  return Status::success();
}

[[nodiscard]] Status
plan_pipeline_cpu_prepared_storage(const PipelineBuildState &build,
                                   PipelineMemoryPlan &plan) {
  plan.cpu_job_slices.assign(build.steps.size(), {});
  plan.cpu_workspace_slices.assign(build.steps.size(), {});
  plan.cpu_alternate_job_slices.clear();
  const bool transactional = !build.state_pairs.empty();
  if (transactional) {
    plan.cpu_alternate_job_slices.assign(build.steps.size(), {});
  }
  if (build.device->backend != Backend::Cpu) {
    return Status::success();
  }
  if (plan.job_owners.size() != build.steps.size() ||
      plan.workspace_routes.size() != build.steps.size() ||
      plan.views.size() != build.steps.size() ||
      plan.cpu_view_layouts.size() != build.steps.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  for (std::size_t index = 0u; index < build.steps.size(); ++index) {
    if (!plan.workspace_routes[index].owns(index)) {
      continue;
    }
    const PipelineBuildStep &step = build.steps[index];
    if (!append_cpu_workspace_slice(plan.cpu_prepared_arena,
                                    step.program->chunks.size(),
                                    plan.cpu_workspace_slices[index])) {
      return Status::fail(Reason::PipelineCapacity);
    }
  }
  for (std::size_t index = 0u; index < build.steps.size(); ++index) {
    if (plan.job_owners[index] != index) {
      continue;
    }
    if (index >= plan.step_resources.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const PipelineStepResourcePlan &sealed = plan.step_resources[index];
    const CpuViewTransferLayout &view = plan.cpu_view_layouts[index];
    if (view.input_count != sealed.inputs.size() ||
        view.output_count != sealed.physical_sources.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const CpuJobBindingCounts counts{
        .inputs = sealed.inputs.size(),
        .outputs = sealed.physical_sources.size(),
        .kernel_views = plan.views[index].size(),
        .input_transfers = view.inputs.size(),
        .output_transfers = view.outputs.size(),
    };
    if (!append_cpu_job_binding_slice(plan.cpu_prepared_arena, counts,
                                      plan.cpu_job_slices[index]) ||
        (transactional &&
         !append_cpu_job_binding_slice(plan.cpu_prepared_arena, counts,
                                       plan.cpu_alternate_job_slices[index]))) {
      return Status::fail(Reason::PipelineCapacity);
    }
  }
  if (build.steps.empty()) {
    return Status::success();
  }
  if (build.device->host_page_bytes == 0u ||
      !seal_cpu_prepared_arena_plan(plan.cpu_prepared_arena,
                                    build.device->host_page_bytes)) {
    return Status::fail(Reason::PipelineCapacity);
  }
  const CpuStorageBytes payload =
      cpu_prepared_arena_payload(plan.cpu_prepared_arena);
  if (!kernel::checked::add(plan.summary.prepared_host_bytes, payload.host,
                            plan.summary.prepared_host_bytes) ||
      !kernel::checked::add(plan.summary.prepared_host_bytes,
                            sizeof(CpuPreparedArena),
                            plan.summary.prepared_host_bytes) ||
      !kernel::checked::add(plan.summary.prepared_tile_bytes, payload.tile,
                            plan.summary.prepared_tile_bytes) ||
      !kernel::checked::add(plan.summary.arena_extent_bytes,
                            plan.cpu_prepared_arena.layout.extent_bytes,
                            plan.summary.arena_extent_bytes)) {
    return Status::fail(Reason::PipelineCapacity);
  }
  return Status::success();
}

[[nodiscard]] Status
plan_pipeline_accel_preparation(const PipelineBuildState &build,
                                PipelineMemoryPlan &plan) {
  plan.accel_preparation = {};
  if (build.device->backend == Backend::Cpu) {
    return Status::success();
  }
  const DeviceOps *const ops = build.device->ops;
  if (ops == nullptr || ops->plan_pipeline_preparation == nullptr ||
      plan.job_owners.size() != build.steps.size() ||
      plan.workspace_routes.size() != build.steps.size() ||
      plan.views.size() != build.steps.size() ||
      plan.window_states.size() != build.steps.size()) {
    return Status::fail(Reason::DeviceInvalid);
  }

  try {
    std::vector<std::uint64_t> entry_counts(build.steps.size(), 0u);
    std::vector<std::uint64_t> occurrence_counts(build.steps.size(), 0u);
    std::vector<std::uint64_t> window_counts(build.steps.size(), 0u);
    std::vector<std::uint64_t> nested_group_counts(build.steps.size(), 0u);
    std::vector<std::uint64_t> map_recurrence_group_counts(build.steps.size(),
                                                           0u);
    std::vector<std::uint64_t> map_recurrence_history_group_counts(
        build.steps.size(), 0u);
    std::vector<std::uint64_t> recurrence_hi(build.steps.size(), 0u);
    std::vector<std::uint64_t> recurrence_lo(build.steps.size(), 0u);
    std::vector<std::uint8_t> active_window_states(build.steps.size(), 0u);
    std::uint64_t window_state_count = 0u;
    std::uint64_t window_descriptor_state_count = 0u;
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

    std::vector<node::accel::detail::PreparedKernelProgramRoute> routes;
    std::vector<
        std::vector<node::accel::detail::PreparedKernelProgramBindingIdentity>>
        route_program_bindings;
    routes.reserve(build.steps.size());
    route_program_bindings.reserve(build.steps.size());
    const std::uint32_t route_copies = build.state_pairs.empty() ? 1u : 2u;
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

    node::accel::detail::PreparedKernelTemplateRegistry templates{};
    std::uint64_t terminal_publication_count = 0u;
    std::uint64_t publication_command_count = 0u;
    for (const PipelinePublicationPlan &publication : plan.publications) {
      const auto *window =
          std::get_if<PipelineWindowPublicationPlan>(&publication);
      const std::uint32_t state = std::visit(
          [](const auto &typed) { return typed.state; }, publication);
      if (state >= plan.window_controls.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const node::accel::detail::NestedTemplateShape *const nested_shape =
          window != nullptr ? pipeline_build_nested_shape(build, state)
                            : nullptr;
      if (window != nullptr &&
          (nested_shape == nullptr || !nested_shape->valid())) {
        return Status::fail(Reason::PipelineInvalid);
      }
      std::uint64_t contribution = 0u;
      if (!node::accel::detail::PreparedKernelPublicationCommandContribution(
              window == nullptr
                  ? node::accel::detail::PreparedKernelPublicationKind::Terminal
                  : node::accel::detail::PreparedKernelPublicationKind::Window,
              nested_shape == nullptr ? 0u : nested_shape->outer_bound(),
              contribution) ||
          !kernel::checked::add(publication_command_count, contribution,
                                publication_command_count) ||
          (window == nullptr &&
           !kernel::checked::add(terminal_publication_count, 1u,
                                 terminal_publication_count))) {
        return Status::fail(Reason::PipelineCapacity);
      }
    }
    plan.accel_preparation = ops->plan_pipeline_preparation(
        *build.device, routes,
        node::accel::detail::PreparedKernelPipelineShape{
            .publication_count = plan.publications.size(),
            .terminal_publication_count = terminal_publication_count,
            .backend_publication_command_count = publication_command_count,
            .window_state_count = window_state_count,
            .window_descriptor_state_count = window_descriptor_state_count,
            .publication_fingerprint_hi = plan.publication_fingerprint_hi,
            .publication_fingerprint_lo = plan.publication_fingerprint_lo,
            .declared_step_count = build.steps.size(),
            .route_copies = route_copies,
            .profile_steps = build.profile == PipelineProfile::Steps,
        },
        templates);
    if (!plan.accel_preparation.ok || !templates.limit.ok ||
        plan.accel_preparation.fingerprint_hi !=
            templates.limit.fingerprint_hi ||
        plan.accel_preparation.fingerprint_lo !=
            templates.limit.fingerprint_lo) {
      return Status::fail(project_reason(plan.accel_preparation.reason,
                                         Reason::LoweringInvalid));
    }
    if (!kernel::checked::add(plan.summary.prepared_host_bytes,
                              plan.accel_preparation.host_bytes,
                              plan.summary.prepared_host_bytes) ||
        !kernel::checked::add(plan.summary.prepared_native_bytes,
                              plan.accel_preparation.native_bytes,
                              plan.summary.prepared_native_bytes)) {
      return Status::fail(Reason::PipelineCapacity);
    }
    return Status::success();
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::PipelineCapacity);
  }
}

[[nodiscard]] Status
plan_pipeline_host_preparation(const PipelineBuildState &build,
                               PipelineMemoryPlan &plan) {
  PipelinePlan &summary = plan.summary;
  const std::size_t step_count = build.steps.size();
  const std::size_t resource_count = plan.hazards.lifetimes.size();
  const bool transactional = !build.state_pairs.empty();
  std::uint64_t host = sizeof(PipelineState) + sizeof(PipelinePublicationState);
  if (!add_extent(host, step_count, sizeof(PipelineStep)) ||
      !add_extent(host, build.logical_step_count, sizeof(PipelineWindow)) ||
      !add_extent(host, resource_count, sizeof(PipelineResource)) ||
      !add_extent(host, plan.view_chunks.size() + plan.scratch.size(),
                  sizeof(std::shared_ptr<BufferState>)) ||
      !add_extent(host, plan.cpu_programs.size(),
                  sizeof(std::shared_ptr<CpuGraphStorage>)) ||
      !add_extent(host, resource_count, sizeof(BufferClaim)) ||
      (transactional &&
       !add_extent(host, resource_count, sizeof(BufferClaim))) ||
      !add_extent(host, plan.publications.size(),
                  sizeof(PipelinePublicationPlan)) ||
      !add_extent(host, build.state_pairs.size(), sizeof(PipelineStatePair)) ||
      !add_extent(host, plan.hazards.dependencies.size(),
                  sizeof(PipelineDependency)) ||
      !add_extent(host, step_count, sizeof(std::uint8_t))) {
    return Status::fail(Reason::PipelineCapacity);
  }
  const bool has_windows = !plan.window_controls.empty();
  if (has_windows &&
      !add_extent(host, step_count + 1u, sizeof(std::uint16_t))) {
    return Status::fail(Reason::PipelineCapacity);
  }
  const auto outputs = pipeline_output_count(plan);
  if (!outputs || !add_extent(host, *outputs, sizeof(PipelineOutputState)) ||
      !add_extent(host, *outputs, sizeof(std::uint32_t))) {
    return Status::fail(outputs ? Reason::PipelineCapacity : outputs.reason());
  }
  if (build.profile == PipelineProfile::Steps) {
    const std::size_t bool_capacity =
        step_count > std::numeric_limits<std::size_t>::max() - 63u
            ? std::numeric_limits<std::size_t>::max()
            : ((step_count + 63u) / 64u) * 64u;
    if (!add_extent(host, 1u, sizeof(PipelineProfileState)) ||
        !add_extent(host, step_count, sizeof(PipelineStepProfile)) ||
        !add_extent(host, step_count, sizeof(std::uint64_t)) ||
        !add_extent(host, bool_capacity, sizeof(bool))) {
      return Status::fail(Reason::PipelineCapacity);
    }
  }

  const std::size_t prepared_buffer_count =
      plan.view_chunks.size() + plan.scratch.size();
  const bool has_arena = prepared_buffer_count != 0u;
  if (has_arena) {
    const std::size_t slot_count = plan.view_slots.size() + plan.scratch.size();
    if (!add_extent(host, 1u, sizeof(JobArena)) ||
        !add_extent(host, prepared_buffer_count,
                    sizeof(std::shared_ptr<BufferState>)) ||
        !add_extent(host, slot_count, sizeof(JobArenaSlot)) ||
        !add_extent(host, plan.scratch.size(),
                    sizeof(node::accel::detail::KernelScratchPage))) {
      return Status::fail(Reason::PipelineCapacity);
    }
    if (slot_count > node::accel::detail::kInlineBindingCapacity &&
        (!add_extent(host, slot_count,
                     sizeof(rund::kernel::ResidentBufferRef)) ||
         !add_extent(host, slot_count, sizeof(std::shared_ptr<void>)))) {
      return Status::fail(Reason::PipelineCapacity);
    }
  }

  if (plan.workspace_routes.size() != step_count ||
      (build.device->backend == Backend::Cpu &&
       plan.cpu_workspace_slices.size() != step_count)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  for (std::size_t index = 0u; index < step_count; ++index) {
    const PipelineBuildStep &step = build.steps[index];
    if (!plan.workspace_routes[index].owns(index)) {
      continue;
    }
    if (build.device->backend != Backend::Cpu &&
        (!add_extent(host, 1u, sizeof(JobWorkspace)) ||
         !add_extent(host, step.program->chunks.size(),
                     sizeof(std::shared_ptr<BufferState>)) ||
         !add_extent(host, step.program->chunks.size(), sizeof(std::size_t)))) {
      return Status::fail(Reason::PipelineCapacity);
    }
  }

  const std::uint64_t copies = transactional ? 2u : 1u;
  for (std::size_t index = 0u; index < step_count; ++index) {
    if (plan.job_owners[index] != index) {
      continue;
    }
    if (index >= plan.step_resources.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const PipelineStepResourcePlan &sealed = plan.step_resources[index];
    std::uint64_t job = sizeof(JobState);
    if (build.device->backend == Backend::Cpu) {
      if (index >= plan.cpu_view_layouts.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const CpuViewTransferLayout &layout = plan.cpu_view_layouts[index];
      if (layout.input_count != sealed.inputs.size() ||
          layout.output_count != sealed.physical_sources.size() ||
          layout.inputs.size() >
              std::numeric_limits<std::size_t>::max() - layout.outputs.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
    } else if (!add_extent(job, sealed.inputs.size(),
                           sizeof(std::shared_ptr<BufferState>)) ||
               !add_extent(job, sealed.inputs.size(), sizeof(JobBufferView)) ||
               !add_extent(job, sealed.physical_sources.size(),
                           sizeof(std::shared_ptr<BufferState>)) ||
               !add_extent(job, sealed.physical_sources.size(),
                           sizeof(JobBufferView)) ||
               !add_extent(job, plan.views[index].size(),
                           sizeof(node::accel::detail::KernelViewSlot))) {
      return Status::fail(Reason::PipelineCapacity);
    }
    std::uint64_t all_jobs = 0u;
    if (!kernel::checked::mul(job, copies, all_jobs) ||
        !kernel::checked::add(host, all_jobs, host)) {
      return Status::fail(Reason::PipelineCapacity);
    }
  }
  if (!kernel::checked::add(summary.prepared_host_bytes, host,
                            summary.prepared_host_bytes)) {
    return Status::fail(Reason::PipelineCapacity);
  }
  return Status::success();
}

} // namespace

[[nodiscard]] Result<std::shared_ptr<const PipelineMemoryPlan>>
plan_memory(const PipelineBuildState &build) {
  try {
    auto result = std::make_shared<PipelineMemoryPlan>();
    std::vector<PipelineFrozenStep> frozen_steps;
    frozen_steps.reserve(build.steps.size());
    for (const PipelineBuildStep &step : build.steps) {
      frozen_steps.push_back(PipelineFrozenStep{
          .program = step.program,
          .logical_step = step.logical_step,
          .iteration = step.iteration,
          .iteration_bound = step.iteration_bound,
          .nested = step.nested,
          .route = step.route,
          .writes_each_iteration = step.writes_each_iteration,
      });
    }
    std::vector<PipelineFrozenNestedWindow> frozen_nested;
    frozen_nested.reserve(build.nested_windows.size());
    for (const PipelineBuildNestedWindow &nested : build.nested_windows) {
      frozen_nested.push_back(PipelineFrozenNestedWindow{
          .shape = nested.shape,
          .recurrent_output_count = nested.recurrent_output_count,
      });
    }
    result->frozen =
        std::make_shared<const PipelineBuildSnapshot>(PipelineBuildSnapshot{
            .device = build.device,
            .steps = std::move(frozen_steps),
            .nested_windows = std::move(frozen_nested),
            .logical_step_count = build.logical_step_count,
            .sealed_repetitions = build.sealed_repetitions,
            .profile = build.profile,
            .commit = build.commit,
        });
    PipelinePlan &summary = result->summary;
    std::unordered_set<const ProgramState *> programs;
    programs.reserve(build.steps.size());
    std::vector<const ProgramState *> unique_programs;
    unique_programs.reserve(build.steps.size());
    std::unordered_map<const ProgramState *, std::size_t> cpu_programs;
    cpu_programs.reserve(build.steps.size());
    result->cpu_storage_by_step.assign(build.steps.size(),
                                       std::numeric_limits<std::size_t>::max());
    result->cpu_route_slices.resize(build.steps.size());
    if (!build.state_pairs.empty()) {
      result->cpu_alternate_route_slices.resize(build.steps.size());
    }
    result->job_owners.resize(build.steps.size());
    for (std::size_t index = 0u; index < build.steps.size(); ++index) {
      const PipelineBuildStep &step = build.steps[index];
      if (step.program == nullptr) {
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineInvalid);
      }
      if (programs.insert(step.program.get()).second) {
        unique_programs.push_back(step.program.get());
        if (!kernel::checked::add(summary.node_count,
                                  static_cast<std::uint64_t>(
                                      step.program->graph_info.nodes.size()),
                                  summary.node_count)) {
          return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
              Reason::PipelineCapacity);
        }
      }
      if (build.device->backend == Backend::Cpu && !step.program->empty() &&
          step.program->cpu_graph != nullptr) {
        const auto [position, inserted] = cpu_programs.emplace(
            step.program.get(), result->cpu_programs.size());
        if (inserted) {
          const auto storage = plan_cpu_graph_storage(step.program);
          const auto route = plan_cpu_run_route(step.program);
          if (!storage || !route) {
            return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
                storage ? route.reason() : storage.reason());
          }
          result->cpu_programs.push_back(step.program);
          result->cpu_storage_plans.push_back(*storage);
          result->cpu_route_plans.push_back(*route);
          if (!merge_cpu_execution_storage_plan(
                  result->cpu_prepared_arena.execution, storage->execution)) {
            return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
                Reason::PipelineCapacity);
          }
          if (!kernel::checked::add(summary.prepared_host_bytes,
                                    storage->private_total.host,
                                    summary.prepared_host_bytes) ||
              !kernel::checked::add(summary.prepared_tile_bytes,
                                    storage->private_total.tile,
                                    summary.prepared_tile_bytes)) {
            return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
                Reason::PipelineCapacity);
          }
        }
        result->cpu_storage_by_step[index] = position->second;
      }
    }

    // Seal resource Views, state ordinals, and Window controls before any
    // private-Job reuse or accounting decision. All post-boundary consumers
    // below read this plan instead of interpreting authored bindings/scalars.
    auto schedule = plan_pipeline_schedule(build, *result);
    if (!schedule) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          schedule.reason(), schedule.location());
    }
    if (result->step_resources.size() != build.steps.size() ||
        result->window_states.size() != build.steps.size() ||
        std::any_of(result->window_states.begin(), result->window_states.end(),
                    [&](const std::uint32_t state) {
                      return state != PipelineResourceUnassigned &&
                             state >= result->window_controls.size();
                    })) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          Reason::PipelineInvalid);
    }

    for (std::size_t index = 0u; index < build.steps.size(); ++index) {
      const PipelineBuildStep &step = build.steps[index];
      std::size_t owner = index;
      const auto control =
          [&](const std::size_t step_index) -> const PipelineWindowControl * {
        const std::uint32_t state = result->window_states[step_index];
        return state == PipelineResourceUnassigned
                   ? nullptr
                   : &result->window_controls[state];
      };
      std::size_t candidate = index;
      if (step.nested != 0u) {
        const std::size_t nested_index = step.nested - 1u;
        if (nested_index >= build.nested_windows.size() ||
            !build.nested_windows[nested_index].shape.retained_owner(
                index, candidate)) {
          return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
              Reason::PipelineInvalid);
        }
      } else if (step.iteration_bound > 1u && step.iteration >= 3u &&
                 index >= 2u) {
        candidate = index - 2u;
      }
      if (candidate < index) {
        const bool same = same_recurrence_phase(
            step, result->step_resources[index], control(index),
            build.steps[candidate], result->step_resources[candidate],
            control(candidate));
        if (step.nested != 0u && !same) {
          return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
              Reason::PipelineInvalid);
        }
        if (same) {
          owner = result->job_owners[candidate];
        }
      }
      result->job_owners[index] = owner;
    }
    const bool transactional = !build.state_pairs.empty();
    for (std::size_t index = 0u; index < build.steps.size(); ++index) {
      if (result->job_owners[index] != index) {
        continue;
      }
      const std::size_t cpu_owner = result->cpu_storage_by_step[index];
      if (cpu_owner == std::numeric_limits<std::size_t>::max()) {
        continue;
      }
      if (cpu_owner >= result->cpu_route_plans.size()) {
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineInvalid);
      }
      const CpuRunRoutePlan &route = result->cpu_route_plans[cpu_owner];
      if (!append_cpu_run_route_slice(result->cpu_prepared_arena, route,
                                      result->cpu_route_slices[index]) ||
          (transactional && !append_cpu_run_route_slice(
                                result->cpu_prepared_arena, route,
                                result->cpu_alternate_route_slices[index]))) {
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineCapacity);
      }
    }
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
          shape.valid() && shape.first() < result->window_states.size();
      const std::uint32_t state = state_range
                                      ? result->window_states[shape.first()]
                                      : PipelineResourceUnassigned;
      const PipelineWindowControl *const control =
          state != PipelineResourceUnassigned &&
                  state < result->window_controls.size()
              ? &result->window_controls[state]
              : nullptr;
      node::accel::detail::NestedTemplateShape expected_shape{};
      const bool window_shape =
          control != nullptr &&
          node::accel::detail::ProveNestedTemplateShape(
              shape.first(), control->maximum, control->tile,
              shape.inner_bound(), expected_shape) &&
          expected_shape == shape;
      if (nested_index >= std::numeric_limits<std::uint16_t>::max() ||
          !shape.valid() || shape.first() < covered_until ||
          shape.end() > build.steps.size() || !window_shape) {
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineInvalid);
      }
      const auto nested_id = static_cast<std::uint16_t>(nested_index + 1u);
      std::uint64_t retained_entries = 0u;
      for (std::size_t index = shape.first(); index < shape.end(); ++index) {
        node::accel::detail::NestedTemplateRouteProjection route{};
        std::size_t retained_candidate = index;
        if (!shape.project(index, route) ||
            !shape.retained_owner(index, retained_candidate) ||
            retained_candidate > index) {
          return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
              Reason::PipelineInvalid);
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
          return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
              Reason::PipelineInvalid);
        }
        const std::size_t expected_owner =
            retained_candidate == index
                ? index
                : result->job_owners[retained_candidate];
        if (result->job_owners[index] != expected_owner) {
          return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
              Reason::PipelineInvalid);
        }
        retained_entries += expected_owner == index ? 1u : 0u;
      }
      if (retained_entries != shape.retained_entry_count()) {
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineInvalid);
      }
      covered_until = shape.end();
      summary.outer_window_count =
          std::max(summary.outer_window_count,
                   static_cast<std::uint64_t>(shape.outer_bound()));
      summary.tile_capacity = std::max(
          summary.tile_capacity, static_cast<std::uint64_t>(control->tile));
      summary.inner_iteration_count =
          std::max(summary.inner_iteration_count,
                   static_cast<std::uint64_t>(shape.inner_bound()));
      if (!kernel::checked::add(nested_commands,
                                shape.authored_occurrence_count(),
                                nested_commands)) {
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineCapacity);
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
      if (!kernel::checked::mul(
              action_logical, static_cast<std::uint64_t>(shape.action_count()),
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
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineCapacity);
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
          return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
              Reason::PipelineInvalid);
        }
        if (!kernel::checked::add(logical_workspace,
                                  step.program->graph_info.memory.logical_bytes,
                                  logical_workspace)) {
          return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
              Reason::PipelineCapacity);
        }
      }
      live_workspace =
          std::max(live_workspace, step.program->graph_info.memory.live_bytes);
    }
    std::size_t internal_resource_count = 0u;
    for (const PipelineResolvedResourcePlan &resource : result->resources) {
      if (const auto *external =
              std::get_if<PipelineExternalResourcePlan>(&resource.locator)) {
        if (external->owner == nullptr ||
            !kernel::checked::add(summary.persistent_bytes, resource.bytes,
                                  summary.persistent_bytes)) {
          return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
              external->owner == nullptr ? Reason::PipelineInvalid
                                         : Reason::PipelineCapacity);
        }
      } else {
        ++internal_resource_count;
        if (!kernel::checked::add(summary.state_bytes, resource.bytes,
                                  summary.state_bytes)) {
          return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
              Reason::PipelineCapacity);
        }
      }
    }
    if (internal_resource_count != build.internals.size()) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          Reason::PipelineInvalid);
    }
    for (const PipelinePublicationPlan &publication : result->publications) {
      const PipelinePublicationTargetPlan &target =
          pipeline_publication_target(publication);
      if (target.view.identity.resource_ordinal >= result->resources.size() ||
          !std::holds_alternative<PipelineExternalResourcePlan>(
              result->resources[target.view.identity.resource_ordinal]
                  .locator)) {
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineInvalid);
      }
      std::uint64_t bytes = 0u;
      std::uint64_t elements = 0u;
      std::uint64_t element_bytes = 0u;
      std::uint64_t occurrences = 1u;
      const std::uint32_t state = std::visit(
          [](const auto &typed) { return typed.state; }, publication);
      if (state >= result->window_controls.size()) {
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineInvalid);
      }
      const PipelineWindowControl &control = result->window_controls[state];
      if (const auto *window =
              std::get_if<PipelineWindowPublicationPlan>(&publication)) {
        elements = control.maximum;
        element_bytes = window->source.identity.element_bytes;
        const node::accel::detail::NestedTemplateShape *const nested_shape =
            pipeline_build_nested_shape(build, state);
        if (nested_shape == nullptr || !nested_shape->valid()) {
          return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
              Reason::PipelineInvalid);
        }
        occurrences = nested_shape->outer_bound();
      } else {
        const auto &terminal =
            std::get<PipelineTerminalPublicationPlan>(publication);
        if (control.final < PipelineWindow::first ||
            control.final > PipelineWindow::second ||
            control.final >= terminal.sources.size()) {
          return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
              Reason::PipelineInvalid);
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
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineCapacity);
      }
    }

    const Status arena =
        plan_pipeline_arena(*build.device, build.steps, *result);
    if (!arena) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          arena.reason());
    }
    const Status views =
        plan_pipeline_views(*build.device, build.steps, *result);
    if (!views) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          views.reason());
    }
    const Status scratch =
        plan_pipeline_scratch(*build.device, unique_programs, *result);
    if (!scratch) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          scratch.reason());
    }
    const Status workspace_routes =
        seal_pipeline_workspace_routes(build, *result);
    if (!workspace_routes) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          workspace_routes.reason());
    }
    const Status cpu_views = plan_pipeline_cpu_views(build, *result);
    if (!cpu_views) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          cpu_views.reason());
    }
    const Status cpu_prepared_storage =
        plan_pipeline_cpu_prepared_storage(build, *result);
    if (!cpu_prepared_storage) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          cpu_prepared_storage.reason());
    }
    const Status accel_preparation =
        plan_pipeline_accel_preparation(build, *result);
    if (!accel_preparation) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          accel_preparation.reason());
    }
    const Status host_preparation =
        plan_pipeline_host_preparation(build, *result);
    if (!host_preparation) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          host_preparation.reason());
    }
    if (!build.state_pairs.empty()) {
      std::uint64_t pairs = 0u;
      if (!kernel::checked::mul(
              static_cast<std::uint64_t>(build.state_pairs.size()),
              static_cast<std::uint64_t>(sizeof(PipelineStatePair)), pairs) ||
          !kernel::checked::add(
              static_cast<std::uint64_t>(sizeof(PipelinePublicationState)),
              pairs, result->publication_committed_bytes)) {
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineCapacity);
      }
    }
    std::uint64_t payload_allocations = 0u;
    if (!kernel::checked::add(
            static_cast<std::uint64_t>(internal_resource_count),
            static_cast<std::uint64_t>(result->chunks.size()),
            payload_allocations) ||
        !kernel::checked::add(
            payload_allocations,
            static_cast<std::uint64_t>(result->view_chunks.size()),
            payload_allocations) ||
        !kernel::checked::add(
            payload_allocations,
            static_cast<std::uint64_t>(result->scratch.size()),
            payload_allocations) ||
        !kernel::checked::add(summary.allocation_count, payload_allocations,
                              summary.allocation_count)) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          Reason::PipelineCapacity);
    }
    std::uint64_t buffer_state_bytes = 0u;
    if (!kernel::checked::mul(summary.allocation_count,
                              static_cast<std::uint64_t>(sizeof(BufferState)),
                              buffer_state_bytes) ||
        !kernel::checked::add(summary.prepared_host_bytes, buffer_state_bytes,
                              summary.prepared_host_bytes)) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          Reason::PipelineCapacity);
    }
    summary.prepared_bytes = summary.prepared_buffer_bytes;
    if (!kernel::checked::add(summary.prepared_bytes,
                              summary.prepared_host_bytes,
                              summary.prepared_bytes) ||
        !kernel::checked::add(summary.prepared_bytes,
                              summary.prepared_tile_bytes,
                              summary.prepared_bytes) ||
        !kernel::checked::add(summary.prepared_bytes,
                              summary.prepared_native_bytes,
                              summary.prepared_bytes)) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          Reason::PipelineCapacity);
    }
    std::uint64_t infrastructure_bytes = 0u;
    if (!kernel::checked::add(summary.state_bytes, summary.prepared_bytes,
                              infrastructure_bytes) ||
        !kernel::checked::add(infrastructure_bytes, summary.transient_bytes,
                              summary.peak_bytes) ||
        !kernel::checked::add(summary.persistent_bytes, summary.peak_bytes,
                              summary.total_bytes)) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          Reason::PipelineCapacity);
    }
    std::uint64_t arena_payload_bytes = 0u;
    std::uint64_t arena_committed_bytes = 0u;
    if (result->cpu_prepared_arena.layout.sealed) {
      const CpuStorageBytes payload =
          cpu_prepared_arena_payload(result->cpu_prepared_arena);
      if (!kernel::checked::add(payload.host, payload.tile,
                                arena_payload_bytes) ||
          !kernel::checked::add(
              arena_committed_bytes,
              result->cpu_prepared_arena.layout.committed_bytes,
              arena_committed_bytes)) {
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineCapacity);
      }
    }
    if (arena_payload_bytes > summary.peak_bytes ||
        !kernel::checked::add(summary.peak_bytes - arena_payload_bytes,
                              arena_committed_bytes,
                              summary.committed_peak_bytes) ||
        result->publication_committed_bytes > summary.committed_peak_bytes) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          Reason::PipelineCapacity);
    }
    std::uint64_t flat_commands = 0u;
    for (std::size_t index = 0u; index < build.steps.size(); ++index) {
      const PipelineBuildStep &step = build.steps[index];
      if (step.route != PipelineRoute::Ordinary) {
        continue;
      }
      ++flat_commands;
    }
    std::uint64_t route_templates = 0u;
    for (std::size_t index = 0u; index < result->job_owners.size(); ++index) {
      if (result->job_owners[index] == index &&
          !kernel::checked::add(route_templates, 1u, route_templates)) {
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineCapacity);
      }
    }
    summary.prepared_template_count = route_templates;
    if (!kernel::checked::add(flat_commands, nested_commands,
                              summary.prepared_command_count)) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          Reason::PipelineCapacity);
    }
    if (!kernel::checked::add(infrastructure_bytes, logical_workspace,
                              summary.logical_bytes) ||
        !kernel::checked::add(infrastructure_bytes, live_workspace,
                              summary.live_bytes) ||
        !kernel::checked::add(infrastructure_bytes, summary.transient_bytes,
                              summary.physical_bytes)) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          Reason::PipelineCapacity);
    }
    if (summary.physical_bytes != summary.peak_bytes) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          Reason::PipelineInvalid);
    }
    return Result<std::shared_ptr<const PipelineMemoryPlan>>::success(
        std::move(result));
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
        Reason::PipelineCapacity);
  }
}

[[nodiscard]] Status materialize_pipeline(PipelineBuildState &build) {
  try {
    if (build.memory == nullptr || build.memory->frozen == nullptr ||
        build.memory->resources.size() !=
            build.memory->hazards.lifetimes.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const PipelineBuildSnapshot &frozen = *build.memory->frozen;
    std::array<std::uint32_t, PipelineIterationCapacity> ordinals{};
    build.materialized_resources.clear();
    build.materialized_resources.reserve(build.memory->resources.size());
    for (const PipelineResolvedResourcePlan &resource :
         build.memory->resources) {
      std::shared_ptr<BufferState> owner;
      bool external_owner = false;
      if (const auto *external =
              std::get_if<PipelineExternalResourcePlan>(&resource.locator)) {
        owner = external->owner;
        external_owner = true;
      } else {
        const auto *internal =
            std::get_if<PipelineInternalResourcePlan>(&resource.locator);
        if (internal == nullptr ||
            resource.count > std::numeric_limits<std::size_t>::max()) {
          return Status::fail(Reason::PipelineInvalid);
        }
        auto made =
            make_input_binding_buffer(build.device, resource.type,
                                      static_cast<std::size_t>(resource.count));
        if (!made) {
          return Status::fail(made.reason());
        }
        owner = std::move(made).value();
        if (internal->fill == PipelineFill::Ordinal) {
          if (resource.type != Type::U32 || resource.count > ordinals.size()) {
            return Status::fail(Reason::PipelineInvalid);
          }
          const auto count = static_cast<std::size_t>(resource.count);
          std::iota(ordinals.begin(), ordinals.begin() + count, 0u);
          WriteStats writes{};
          const Status written = write_buffer(owner,
                                              HostView{.data = ordinals.data(),
                                                       .count = count,
                                                       .type = Type::U32},
                                              writes);
          if (!written) {
            return written;
          }
        }
      }
      if (owner == nullptr) {
        return Status::fail(Reason::BindingInvalid);
      }
      // External semantic validation remains prepare admission's authority so
      // it preserves the public BindingDeviceMismatch/type/shape reasons.
      // Materialization only retains that already-sealed owner. Internals are
      // created here and therefore must exactly match this constructor's plan.
      if (!external_owner &&
          (owner->device != frozen.device || owner->type != resource.type ||
           owner->count != resource.count || owner->bytes != resource.bytes ||
           owner->physical_bytes < owner->bytes)) {
        return Status::fail(Reason::PipelineInvalid);
      }
      build.materialized_resources.push_back(owner);
    }
    return Status::success();
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::PipelineCapacity);
  }
}

[[nodiscard]] Result<PipelineMemorySet>
make_pipeline_memory(const std::shared_ptr<DeviceState> &device,
                     const std::span<const PipelineFrozenStep> steps,
                     const PipelineMemoryPlan &plan) {
  try {
    PipelineMemorySet result;
    result.steps.resize(steps.size());
    if (plan.cpu_storage_by_step.size() != steps.size() ||
        plan.workspace_routes.size() != steps.size() ||
        plan.cpu_route_slices.size() != steps.size() ||
        (device->backend == Backend::Cpu &&
         (plan.cpu_job_slices.size() != steps.size() ||
          plan.cpu_workspace_slices.size() != steps.size())) ||
        (!plan.cpu_alternate_job_slices.empty() &&
         plan.cpu_alternate_job_slices.size() != steps.size()) ||
        (!plan.cpu_alternate_route_slices.empty() &&
         plan.cpu_alternate_route_slices.size() != steps.size())) {
      return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
    }
    result.cpu_storage_by_step = plan.cpu_storage_by_step;
    if (plan.cpu_storage_plans.size() != plan.cpu_programs.size() ||
        plan.cpu_route_plans.size() != plan.cpu_programs.size()) {
      return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
    }
    const bool has_cpu_prepared_jobs =
        device->backend == Backend::Cpu && !steps.empty();
    if (has_cpu_prepared_jobs != plan.cpu_prepared_arena.layout.sealed) {
      return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
    }
    if (has_cpu_prepared_jobs) {
      auto prepared = make_cpu_prepared_arena(plan.cpu_prepared_arena);
      if (!prepared) {
        return Result<PipelineMemorySet>::fail(prepared.reason());
      }
      result.cpu_prepared_arena = std::move(prepared).value();
    }
    result.cpu_storage.reserve(plan.cpu_programs.size());
    for (std::size_t index = 0u; index < plan.cpu_programs.size(); ++index) {
      auto storage = make_cpu_graph_storage(plan.cpu_programs[index],
                                            plan.cpu_storage_plans[index],
                                            result.cpu_prepared_arena);
      if (!storage || storage.value() == nullptr) {
        return Result<PipelineMemorySet>::fail(
            storage ? Reason::CpuRuntimeInvalid : storage.reason());
      }
      result.cpu_storage.push_back(std::move(storage).value());
    }
    if (result.cpu_prepared_arena != nullptr &&
        !result.cpu_prepared_arena->claims_complete(
            plan.cpu_prepared_arena.execution)) {
      return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
    }
    if (plan.steps.size() != steps.size() + 1u || plan.steps.empty() ||
        plan.steps.front() != 0u || plan.steps.back() != plan.offsets.size() ||
        plan.owners.size() != plan.offsets.size()) {
      return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
    }
    result.buffers.reserve(plan.chunks.size());
    for (const std::size_t count : plan.chunks) {
      if (count == 0u) {
        return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
      }
      auto made = make_workspace_buffer(device, count);
      if (!made) {
        return Result<PipelineMemorySet>::fail(made.reason());
      }
      result.buffers.push_back(std::move(made).value());
    }
    if (plan.views.size() != steps.size()) {
      return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
    }
    result.prepared.reserve(plan.view_chunks.size() + plan.scratch.size());
    if (!plan.view_chunks.empty() || !plan.scratch.empty()) {
      result.arena = std::make_shared<JobArena>();
      result.arena->buffers.reserve(plan.view_chunks.size() +
                                    plan.scratch.size());
      result.arena->slots.reserve(plan.view_slots.size() + plan.scratch.size());
      result.arena->scratch.reserve(plan.scratch.size());
    }
    for (const std::size_t words : plan.view_chunks) {
      if (words == 0u) {
        return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
      }
      auto made = make_workspace_buffer(device, words);
      if (!made) {
        return Result<PipelineMemorySet>::fail(made.reason());
      }
      result.prepared.push_back(std::move(made).value());
      result.arena->buffers.push_back(result.prepared.back());
    }
    for (const PipelineMemoryPlan::ViewSlot slot : plan.view_slots) {
      if (slot.words == 0u || slot.owner >= plan.view_chunks.size() ||
          slot.offset_words > plan.view_chunks[slot.owner] ||
          slot.words > plan.view_chunks[slot.owner] - slot.offset_words) {
        return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
      }
      result.arena->slots.push_back(JobArenaSlot{
          .words = slot.words,
          .owner = slot.owner,
          .offset_words = slot.offset_words,
      });
    }
    for (std::size_t index = 0u; index < plan.scratch.size(); ++index) {
      const node::accel::detail::KernelScratchPage scratch =
          plan.scratch[index];
      if (scratch.bytes == 0u || scratch.bytes % memory::Word != 0u ||
          scratch.bytes / memory::Word >
              static_cast<std::uint64_t>(
                  std::numeric_limits<std::size_t>::max()) ||
          scratch.slot != plan.view_slots.size() + index ||
          result.arena == nullptr) {
        return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
      }
      const std::size_t words =
          static_cast<std::size_t>(scratch.bytes / memory::Word);
      auto made = make_workspace_buffer(device, words);
      if (!made) {
        return Result<PipelineMemorySet>::fail(made.reason());
      }
      result.prepared.push_back(std::move(made).value());
      result.arena->buffers.push_back(result.prepared.back());
      const std::size_t slot = result.arena->slots.size();
      if (slot != scratch.slot) {
        return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
      }
      result.arena->slots.push_back(
          JobArenaSlot{.words = words,
                       .owner = result.arena->buffers.size() - 1u,
                       .offset_words = 0u});
      result.arena->scratch.push_back(scratch);
    }
    for (std::size_t step_index = 0u; step_index < steps.size(); ++step_index) {
      const PipelineFrozenStep &step = steps[step_index];
      const auto &chunks = step.program->chunks;
      const PipelineWorkspaceRoute workspace_route =
          plan.workspace_routes[step_index];
      if (!workspace_route.present()) {
        continue;
      }
      const std::size_t workspace_owner = workspace_route.owner;
      if (workspace_owner != step_index) {
        if (workspace_owner >= step_index ||
            !plan.workspace_routes[workspace_owner].owns(workspace_owner) ||
            result.steps[workspace_owner] == nullptr ||
            result.steps[workspace_owner]->program != step.program) {
          return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
        }
        result.steps[step_index] = result.steps[workspace_owner];
        continue;
      }
      const std::vector<std::uint32_t> &order = step.program->chunk_order;
      if (order.size() != chunks.size()) {
        return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
      }
      std::shared_ptr<JobWorkspace> workspace;
      if (device->backend == Backend::Cpu) {
        if (result.cpu_prepared_arena == nullptr) {
          return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
        }
        CpuWorkspaceStorage storage{};
        if (!result.cpu_prepared_arena->view(
                plan.cpu_workspace_slices[step_index], storage) ||
            storage.workspace == nullptr ||
            !storage.workspace->buffers.bind(storage.buffers, chunks.size()) ||
            !storage.workspace->offsets.bind(storage.offsets, chunks.size())) {
          return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
        }
        workspace = std::shared_ptr<JobWorkspace>(result.cpu_prepared_arena,
                                                  storage.workspace);
      } else {
        workspace = std::make_shared<JobWorkspace>();
        workspace->buffers.resize(chunks.size());
        workspace->offsets.resize(chunks.size());
      }
      workspace->program = step.program;
      workspace->arena = result.arena;
      const std::size_t begin = plan.steps[step_index];
      const std::size_t end = plan.steps[step_index + 1u];
      if (begin > end || end > plan.offsets.size() ||
          end - begin != order.size()) {
        return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
      }
      for (std::size_t rank = 0u; rank < order.size(); ++rank) {
        const std::size_t chunk = order[rank];
        const std::size_t owner = plan.owners[begin + rank];
        const std::size_t offset = plan.offsets[begin + rank];
        if (chunk >= chunks.size() || owner >= result.buffers.size() ||
            owner >= plan.chunks.size() || offset > plan.chunks[owner] ||
            chunks[chunk].count > plan.chunks[owner] - offset) {
          return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
        }
        workspace->buffers[chunk] = result.buffers[owner];
        workspace->offsets[chunk] = offset;
      }
      result.steps[step_index] = std::move(workspace);
    }
    return Result<PipelineMemorySet>::success(std::move(result));
  } catch (const std::bad_alloc &) {
    return Result<PipelineMemorySet>::fail(Reason::BufferCapacity);
  }
}

} // namespace rund::compute::detail
