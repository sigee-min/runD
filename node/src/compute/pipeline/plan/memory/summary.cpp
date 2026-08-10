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
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rund::compute::detail {

[[nodiscard]] Status
finalize_pipeline_plan(const PipelineBuildState &build,
                       PipelineMemoryPlan &plan,
                       const PipelinePlanningModel &model,
                       const PipelinePlanningTotals &workload) {
  PipelinePlan &summary = plan.summary;
  const Status arena = plan_pipeline_arena(*build.device, build.steps, plan);
  if (!arena) {
    return Status::fail(arena.reason());
  }
  const Status views = plan_pipeline_views(*build.device, build.steps, plan);
  if (!views) {
    return Status::fail(views.reason());
  }
  const Status scratch =
      plan_pipeline_scratch(*build.device, model.unique_programs, plan);
  if (!scratch) {
    return Status::fail(scratch.reason());
  }
  const Status workspace_routes = seal_pipeline_workspace_routes(build, plan);
  if (!workspace_routes) {
    return Status::fail(workspace_routes.reason());
  }
  const Status cpu_views = plan_pipeline_cpu_views(build, plan);
  if (!cpu_views) {
    return Status::fail(cpu_views.reason());
  }
  const Status cpu_prepared_storage =
      plan_pipeline_cpu_prepared_storage(build, plan);
  if (!cpu_prepared_storage) {
    return Status::fail(cpu_prepared_storage.reason());
  }
  const Status accel_preparation = plan_pipeline_accel_preparation(build, plan);
  if (!accel_preparation) {
    return Status::fail(accel_preparation.reason());
  }
  const Status host_preparation = plan_pipeline_host_preparation(build, plan);
  if (!host_preparation) {
    return Status::fail(host_preparation.reason());
  }
  if (!build.state_pairs.empty()) {
    std::uint64_t pairs = 0u;
    if (!kernel::checked::mul(
            static_cast<std::uint64_t>(build.state_pairs.size()),
            static_cast<std::uint64_t>(sizeof(PipelineStatePair)), pairs) ||
        !kernel::checked::add(
            static_cast<std::uint64_t>(sizeof(PipelinePublicationState)), pairs,
            plan.publication_committed_bytes)) {
      return Status::fail(Reason::PipelineCapacity);
    }
  }
  std::uint64_t payload_allocations = 0u;
  if (!kernel::checked::add(
          static_cast<std::uint64_t>(workload.internal_resource_count),
          static_cast<std::uint64_t>(plan.chunks.size()),
          payload_allocations) ||
      !kernel::checked::add(payload_allocations,
                            static_cast<std::uint64_t>(plan.view_chunks.size()),
                            payload_allocations) ||
      !kernel::checked::add(payload_allocations,
                            static_cast<std::uint64_t>(plan.scratch.size()),
                            payload_allocations) ||
      !kernel::checked::add(summary.allocation_count, payload_allocations,
                            summary.allocation_count)) {
    return Status::fail(Reason::PipelineCapacity);
  }
  std::uint64_t buffer_state_bytes = 0u;
  if (!kernel::checked::mul(summary.allocation_count,
                            static_cast<std::uint64_t>(sizeof(BufferState)),
                            buffer_state_bytes) ||
      !kernel::checked::add(summary.prepared_host_bytes, buffer_state_bytes,
                            summary.prepared_host_bytes)) {
    return Status::fail(Reason::PipelineCapacity);
  }
  if (build.residency.pages != nullptr) {
    const residency::ResidencyPlan &pages = *build.residency.pages;
    const residency::LinearPlan &linear = pages.linear();
    std::uint64_t combined_page_bytes = 0u;
    if (!pages.identity() || build.residency.logical_bytes == 0u ||
        build.residency.input_page_bytes == 0u ||
        build.residency.output_page_bytes == 0u ||
        !kernel::checked::add(build.residency.input_page_bytes,
                              build.residency.output_page_bytes,
                              combined_page_bytes) ||
        combined_page_bytes != pages.page_bytes() ||
        build.residency.staging_bytes == 0u ||
        build.residency.resident_bytes == 0u ||
        build.residency.slot_count == 0u ||
        build.residency.slot_count != linear.slot_capacity() ||
        build.residency.first_step > build.steps.size() ||
        build.residency.slot_count >
            build.steps.size() - build.residency.first_step ||
        !kernel::checked::add(summary.prepared_host_bytes,
                              build.residency.staging_bytes,
                              summary.prepared_host_bytes) ||
        !kernel::checked::add(summary.allocation_count, 1u,
                              summary.allocation_count)) {
      return Status::fail(Reason::PipelineCapacity);
    }
    plan.residency = build.residency;
    if (build.device->backend == Backend::Vulkan) {
      std::uint64_t input_arena_bytes = 0u;
      std::uint64_t output_arena_bytes = 0u;
      if (build.device->ops == nullptr ||
          build.device->ops->pipeline_transfer_storage_bytes == nullptr ||
          !kernel::checked::mul(build.residency.slot_count,
                                build.residency.input_page_bytes,
                                input_arena_bytes) ||
          !kernel::checked::mul(build.residency.slot_count,
                                build.residency.output_page_bytes,
                                output_arena_bytes)) {
        return Status::fail(Reason::PipelineCapacity);
      }
      const std::uint64_t transfer_bytes =
          build.device->ops->pipeline_transfer_storage_bytes(
              *build.device, std::max(input_arena_bytes, output_arena_bytes));
      if (transfer_bytes == 0u ||
          !kernel::checked::add(summary.prepared_native_bytes, transfer_bytes,
                                summary.prepared_native_bytes) ||
          !kernel::checked::add(summary.allocation_count, 1u,
                                summary.allocation_count)) {
        return Status::fail(Reason::PipelineCapacity);
      }
      plan.residency.transfer_committed_bytes = transfer_bytes;
    }
    summary.residency.logical_bytes = build.residency.logical_bytes;
    summary.residency.page_bytes = pages.page_bytes();
    summary.residency.page_count = linear.page_count();
    summary.residency.slot_capacity = linear.slot_capacity();
    summary.residency.working_set_bytes = build.residency.resident_bytes;
    summary.residency.wave_count = linear.wave_count();
    summary.residency.identity_hi = pages.identity().hi;
    summary.residency.identity_lo = pages.identity().lo;
  }
  summary.prepared_bytes = summary.prepared_buffer_bytes;
  if (!kernel::checked::add(summary.prepared_bytes, summary.prepared_host_bytes,
                            summary.prepared_bytes) ||
      !kernel::checked::add(summary.prepared_bytes, summary.prepared_tile_bytes,
                            summary.prepared_bytes) ||
      !kernel::checked::add(summary.prepared_bytes,
                            summary.prepared_native_bytes,
                            summary.prepared_bytes)) {
    return Status::fail(Reason::PipelineCapacity);
  }
  std::uint64_t infrastructure_bytes = 0u;
  if (!kernel::checked::add(summary.state_bytes, summary.prepared_bytes,
                            infrastructure_bytes) ||
      !kernel::checked::add(infrastructure_bytes, summary.transient_bytes,
                            summary.peak_bytes) ||
      !kernel::checked::add(summary.persistent_bytes, summary.peak_bytes,
                            summary.total_bytes)) {
    return Status::fail(Reason::PipelineCapacity);
  }
  std::uint64_t arena_payload_bytes = 0u;
  std::uint64_t arena_committed_bytes = 0u;
  if (plan.cpu_prepared_arena.layout.sealed) {
    const CpuStorageBytes payload =
        cpu_prepared_arena_payload(plan.cpu_prepared_arena);
    if (!kernel::checked::add(payload.host, payload.tile,
                              arena_payload_bytes) ||
        !kernel::checked::add(arena_committed_bytes,
                              plan.cpu_prepared_arena.layout.committed_bytes,
                              arena_committed_bytes)) {
      return Status::fail(Reason::PipelineCapacity);
    }
  }
  const auto buffers = plan_buffer_commitment(*build.device, plan);
  if (!buffers || buffers->logical > summary.peak_bytes) {
    return Status::fail(buffers ? Reason::PipelineCapacity : buffers.reason());
  }
  std::uint64_t committed_without_arena = 0u;
  if (!kernel::checked::add(summary.peak_bytes - buffers->logical,
                            buffers->committed, committed_without_arena) ||
      arena_payload_bytes > committed_without_arena ||
      !kernel::checked::add(committed_without_arena - arena_payload_bytes,
                            arena_committed_bytes,
                            summary.committed_peak_bytes) ||
      plan.publication_committed_bytes > summary.committed_peak_bytes) {
    return Status::fail(Reason::PipelineCapacity);
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
  for (std::size_t index = 0u; index < plan.job_owners.size(); ++index) {
    if (plan.job_owners[index] == index &&
        !kernel::checked::add(route_templates, 1u, route_templates)) {
      return Status::fail(Reason::PipelineCapacity);
    }
  }
  summary.prepared_template_count = route_templates;
  if (!kernel::checked::add(flat_commands, workload.nested_commands,
                            summary.prepared_command_count)) {
    return Status::fail(Reason::PipelineCapacity);
  }
  if (!kernel::checked::add(infrastructure_bytes, workload.logical_workspace,
                            summary.logical_bytes) ||
      !kernel::checked::add(infrastructure_bytes, workload.live_workspace,
                            summary.live_bytes) ||
      !kernel::checked::add(infrastructure_bytes, summary.transient_bytes,
                            summary.physical_bytes)) {
    return Status::fail(Reason::PipelineCapacity);
  }
  if (summary.physical_bytes != summary.peak_bytes) {
    return Status::fail(Reason::PipelineInvalid);
  }
  return Status::success();
}

} // namespace rund::compute::detail
