#include "arena.hpp"

#include "../../backend.hpp"
#include "../../device/state.hpp"
#include "../../memory/arena.hpp"
#include "../../status.hpp"
#include "../state.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <limits>

namespace rund::compute::detail {

Status plan_pipeline_scratch(const DeviceState &device,
                             const std::span<const PipelineBuildStep> steps,
                             PipelineMemoryPlan &plan) {
  plan.scratch_chunks.clear();
  plan.summary.scratch_bytes = 0u;
  plan.summary.scratch_count = 0u;
  if (device.backend == Backend::Cpu) {
    return Status::success();
  }
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr || device.ops == nullptr ||
      device.ops->plan_scratch == nullptr ||
      !kernel::ComputeStorageAlignmentValid(
          accel->pick.caps.storage_alignment)) {
    return Status::fail(Reason::DeviceInvalid);
  }
  const std::uint64_t page = memory::arena_bytes(device);
  if (page == 0u || page % memory::Word != 0u ||
      page % accel->pick.caps.storage_alignment != 0u) {
    return Status::fail(Reason::PipelineCapacity);
  }
  std::size_t page_count = 0u;
  std::uint64_t last_bytes = 0u;
  for (const PipelineBuildStep &step : steps) {
    if (step.program == nullptr) {
      return Status::fail(Reason::PipelineInvalid);
    }
    if (step.program->accel == nullptr) {
      if (step.program->graph_info.nodes.empty()) {
        continue;
      }
      return Status::fail(Reason::PipelineInvalid);
    }
    const node::accel::detail::KernelScratchPlan scratch =
        device.ops->plan_scratch(device, step.program->accel->kernel,
                                 accel->pick.caps.storage_alignment, page);
    if (!scratch.ok) {
      return Status::fail(
          project_reason(scratch.reason, Reason::LoweringInvalid));
    }
    if (scratch.page_count > page_count) {
      page_count = scratch.page_count;
      last_bytes = scratch.last_bytes;
    } else if (scratch.page_count == page_count) {
      last_bytes = std::max(last_bytes, scratch.last_bytes);
    }
  }
  if (page_count == 0u) {
    return Status::success();
  }
  for (std::size_t index = 0u; index < page_count; ++index) {
    const std::uint64_t bytes = index + 1u == page_count ? last_bytes : page;
    if (bytes % memory::Word != 0u ||
        bytes / memory::Word > static_cast<std::uint64_t>(
                                   std::numeric_limits<std::size_t>::max())) {
      return Status::fail(Reason::PipelineCapacity);
    }
    plan.scratch_chunks.push_back(
        static_cast<std::size_t>(bytes / memory::Word));
  }
  std::uint64_t physical = 0u;
  for (const std::size_t words : plan.scratch_chunks) {
    std::uint64_t bytes = 0u;
    if (!kernel::checked::mul(static_cast<std::uint64_t>(words), memory::Word,
                              bytes) ||
        !kernel::checked::add(physical, bytes, physical)) {
      return Status::fail(Reason::PipelineCapacity);
    }
  }
  if (!kernel::checked::add(plan.summary.prepared_bytes, physical,
                            plan.summary.prepared_bytes)) {
    return Status::fail(Reason::PipelineCapacity);
  }
  plan.summary.scratch_bytes = physical;
  plan.summary.scratch_count = plan.scratch_chunks.size();
  return Status::success();
}

} // namespace rund::compute::detail
