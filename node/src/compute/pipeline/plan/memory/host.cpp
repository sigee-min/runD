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
} // namespace

[[nodiscard]] Status
plan_pipeline_host_preparation(const PipelineBuildState &build,
                               PipelineMemoryPlan &plan) {
  PipelinePlan &summary = plan.summary;
  const std::size_t step_count = build.steps.size();
  const std::size_t resource_count = plan.hazards.lifetimes.size();
  const bool transactional = !build.state_pairs.empty();
  std::uint64_t host = sizeof(PipelineState) + sizeof(PipelinePublicationState);
  if (!add_extent(host, step_count, sizeof(PipelineStep)) ||
      !add_extent(host, build.logical_step_count, PipelineWindows::entry_bytes()) ||
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
  std::size_t job_count = 0u;
  for (std::size_t index = 0u; index < plan.job_owners.size(); ++index) {
    job_count += plan.job_owners[index] == index ? 1u : 0u;
  }
  if (!add_extent(host, job_count,
                  sizeof(PipelineMemoryOwner) * (transactional ? 2u : 1u))) {
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
} // namespace rund::compute::detail
