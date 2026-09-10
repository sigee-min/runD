#include "../../state/assembly.hpp"
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
} // namespace

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
} // namespace rund::compute::detail
