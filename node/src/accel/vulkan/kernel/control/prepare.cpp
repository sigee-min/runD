#include "model.hpp"
#include "source.hpp"

#include "../lease.hpp"
#include "../pipeline/source_artifact.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#include "../../buffer/create/telemetry.hpp"
#include "../../collective/pipeline.hpp"
#include "../../descriptor.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace rund::node::accel::detail {
namespace {

void ReleaseDescriptorLeases(VulkanPipelineControlResources &control) {
  for (const VulkanCollectiveDescriptorLease lease :
       control.descriptor_leases) {
    if (lease.pipeline != nullptr &&
        lease.slot < lease.pipeline->descriptor_leased.size()) {
      lease.pipeline->descriptor_leased[lease.slot] = false;
    }
  }
  control.descriptor_leases.clear();
}

[[nodiscard]] bool ValidSource(const VulkanPipelineStatusSource &source) {
  if (source.raw == nullptr || source.raw->buffer == VK_NULL_HANDLE ||
      source.count == 0u || source.mapping_count == 0u ||
      source.mapping_count > source.raw_values.size() ||
      (source.offset & (sizeof(std::uint32_t) - 1u)) != 0u ||
      source.offset > source.raw->bytes ||
      static_cast<std::uint64_t>(source.count) * sizeof(std::uint32_t) >
          source.raw->bytes - source.offset ||
      source.rule == VulkanPipelineStatusRule::None) {
    return false;
  }
  for (std::size_t index = 0u; index < source.mapping_count; ++index) {
    if (!CanonicalReasonStatus(source.reasons[index]) ||
        source.reasons[index] == 0u) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] PreparedMemory ExactMemory(const std::uint64_t bytes) noexcept {
  return PreparedMemory{
      .current = bytes, .peak = bytes, .cumulative = bytes, .budget = bytes};
}

} // namespace

rund::AccelCheck PrepareVulkanPipelineControl(
    VulkanAdapter &adapter,
    const std::span<VulkanPipelineCanonicalStatus> statuses,
    const PreparedPipelineStatusLayout &layout,
    const std::size_t telemetry_count, const bool profile_steps,
    VulkanPipelineControlResources &control, PreparedPipelineMemory &memory) {
  control = {};
  control.adapter = &adapter;
  memory.device = {};
  memory.staging = {};
  const std::uint64_t arena_bytes =
      static_cast<std::uint64_t>(layout.status_entry_count) *
      sizeof(std::uint32_t);
  const std::uint64_t profile_bytes =
      profile_steps ? static_cast<std::uint64_t>(layout.declared_step_count) *
                          PreparedPipelineStepControlBytes
                    : 0u;
  if ((arena_bytes != 0u &&
       !CreateVulkanBuffer(adapter, arena_bytes,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, control.arena,
                           nullptr, VulkanMemoryUse::Device)) ||
      !CreateVulkanBuffer(adapter, sizeof(PreparedPipelineControl),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          control.summary) ||
      (profile_bytes != 0u &&
       !CreateVulkanBuffer(adapter, profile_bytes,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           control.profile))) {
    DestroyVulkanPipelineControl(control);
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }
  if (!ClearVulkanBuffer(control.summary, sizeof(PreparedPipelineControl))) {
    DestroyVulkanPipelineControl(control);
    return rund::AccelCheck{false, "accel_vulkan_memory_unavailable"};
  }
  control.profile_step_count = profile_steps ? layout.declared_step_count : 0u;
  control.declared_step_count = layout.declared_step_count;
  control.generation_stride = layout.generation_stride;

  if (statuses.size() >
      std::numeric_limits<std::size_t>::max() - 1u - telemetry_count) {
    DestroyVulkanPipelineControl(control);
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  control.descriptor_leases.reserve(statuses.size() + 1u + telemetry_count);
  bool ready = false;
  {
    VulkanLeaseScope lease_scope{adapter, control.descriptor_leases};
    const std::string_view canonical_source = VulkanCanonicalStatusSourceText();
    if (!statuses.empty()) {
      const auto canonical_plan = VulkanPipelineControlPlan(canonical_source);
      const auto canonical_artifact =
          VulkanFixedSourceArtifact(canonical_source);
      if (canonical_artifact.ok) {
        control.canonical_pipeline = AcquireVulkanCollectivePipeline(
            adapter, 2u, sizeof(VulkanCanonicalStatusParams), canonical_plan,
            canonical_artifact);
      }
    }
    const std::string_view reduce_source = VulkanReduceStatusSourceText();
    const auto reduce_plan = VulkanPipelineControlPlan(reduce_source);
    const auto reduce_artifact = VulkanFixedSourceArtifact(reduce_source);
    constexpr std::uint32_t reduce_descriptor_count = 2u;
    if (reduce_artifact.ok) {
      control.reduce_pipeline = AcquireVulkanCollectivePipeline(
          adapter, reduce_descriptor_count, sizeof(VulkanPipelineControlParams),
          reduce_plan, reduce_artifact);
    }
    ready = control.reduce_pipeline != nullptr &&
            (statuses.empty() || control.canonical_pipeline != nullptr);
    ready = ready &&
            (statuses.empty() ||
             ReserveVulkanCollectiveDescriptorDemand(
                 adapter, *control.canonical_pipeline, 2u, statuses.size())) &&
            ReserveVulkanCollectiveDescriptorDemand(
                adapter, *control.reduce_pipeline, reduce_descriptor_count, 1u);
    std::uint32_t expected_first = 0u;
    for (std::size_t index = 0u; ready && index < statuses.size(); ++index) {
      VulkanPipelineCanonicalStatus &status = statuses[index];
      const std::uint64_t groups =
          (static_cast<std::uint64_t>(status.source.count) +
           VulkanPipelineControlThreads - 1u) /
          VulkanPipelineControlThreads;
      ready = status.first == expected_first && ValidSource(status.source) &&
              status.source.count <=
                  std::numeric_limits<std::uint32_t>::max() - expected_first &&
              groups != 0u && groups <= adapter.max_dispatch_groups &&
              AcquireVulkanCollectiveDescriptorSet(
                  adapter, *control.canonical_pipeline, 2u, status.descriptor);
      if (ready) {
        const std::array<VulkanStorageBinding, 2u> buffers{
            VulkanStorageBinding{
                .buffer = status.source.raw,
                .offset = status.source.offset,
                .range = static_cast<VkDeviceSize>(status.source.count) *
                         sizeof(std::uint32_t)},
            VulkanStorageBindingFor(control.arena)};
        ready = WriteVulkanStorageDescriptorSet(adapter, status.descriptor,
                                                buffers);
        expected_first += status.source.count;
      }
    }
    ready = ready && expected_first == layout.status_entry_count;
    ready = ready && AcquireVulkanCollectiveDescriptorSet(
                         adapter, *control.reduce_pipeline,
                         reduce_descriptor_count, control.reduce_descriptor);
    if (ready) {
      const VulkanBuffer *const arena =
          layout.status_entry_count == 0u ? &control.summary : &control.arena;
      const std::array<const VulkanBuffer *, 2u> buffers{arena,
                                                         &control.summary};
      ready = WriteVulkanStorageDescriptorSet(
          adapter, control.reduce_descriptor, buffers);
    }
  }
  if (!ready) {
    const char *const reason = VulkanLastError(&adapter);
    DestroyVulkanPipelineControl(control);
    return rund::AccelCheck{false, reason == nullptr || reason[0] == '\0'
                                       ? "accel_vulkan_descriptor_unavailable"
                                       : reason};
  }
  memory.device = ExactMemory(control.arena.bytes);
  memory.staging = ExactMemory(control.summary.bytes + control.profile.bytes);
  return rund::AccelCheck{true, "ok"};
}

void DestroyVulkanPipelineControl(
    VulkanPipelineControlResources &control) noexcept {
  if (control.adapter != nullptr) {
    ReleaseDescriptorLeases(control);
    ReleaseVulkanBuffer(*control.adapter, control.arena);
    ReleaseVulkanBuffer(*control.adapter, control.summary);
    ReleaseVulkanBuffer(*control.adapter, control.profile);
  }
  control = {};
}

} // namespace rund::node::accel::detail

#endif
