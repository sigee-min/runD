#include "../../adapter/error.hpp"

#include "local.hpp"

#include "../../../kernel/reset/projection.hpp"
#include "../../../kernel/reset/proof.hpp"
#include "../../buffer/resident/find.hpp"
#include "../../collective/pipeline.hpp"
#include "../../descriptor.hpp"
#include "../../resident/access.hpp"
#include "../reset.hpp"
#include "../reset/source.hpp"

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/model.hpp>
#include <rund/counter.hpp>

#include <array>
#include <cstddef>
#include <limits>
#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool ResetBinding(const VulkanAdapter &adapter,
                                const VulkanReset &clear,
                                VulkanStorageBinding &binding,
                                VkDeviceSize &origin) noexcept {
  if (clear.resident.device_buffer == nullptr || adapter.storage_align == 0u ||
      !clear.range.valid()) {
    return false;
  }
  const std::uint64_t base =
      clear.range.offset() - clear.range.offset() % adapter.storage_align;
  const std::uint64_t bytes = clear.range.end() - base;
  if (bytes == 0u || bytes > adapter.storage_limit ||
      base > std::numeric_limits<VkDeviceSize>::max() ||
      bytes > std::numeric_limits<VkDeviceSize>::max()) {
    return false;
  }
  binding = VulkanStorageBinding{
      .buffer = clear.resident.device_buffer,
      .offset = static_cast<VkDeviceSize>(base),
      .range = static_cast<VkDeviceSize>(bytes),
  };
  origin = static_cast<VkDeviceSize>(base);
  return true;
}

} // namespace

[[nodiscard]] rund::AccelCheck
PrepareVulkanResetCommands(VulkanAdapter &adapter,
                           VulkanKernelResources &resources) {
  const bool captured = IsPipelinePrivatePreparation(resources.mode);
  bool needs_pipeline = false;
  std::uint64_t descriptor_set_count = 0u;
  for (const VulkanReset &clear : resources.resets) {
    needs_pipeline = needs_pipeline || clear.shader;
    if (clear.shader && !rund::kernel::checked::add(descriptor_set_count, 1u,
                                                    descriptor_set_count)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  if (!needs_pipeline) {
    return rund::AccelCheck{true, "ok"};
  }
  if (captured) {
    resources.reset_pipeline = resources.program == nullptr
                                   ? nullptr
                                   : resources.program->reset_pipeline;
  } else {
    const rund::kernel::LoweringArtifact artifact = VulkanResetArtifact();
    if (!artifact.ok) {
      return rund::AccelCheck{false, artifact.reason};
    }
    resources.reset_pipeline = AcquireVulkanCollectivePipeline(
        adapter, 1u, sizeof(reset::Params), VulkanResetPlan(), artifact);
  }
  if (resources.reset_pipeline == nullptr) {
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }
  if (!captured &&
      !ReserveVulkanCollectiveDescriptorDemand(
          adapter, *resources.reset_pipeline, 1u, descriptor_set_count)) {
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }
  for (VulkanReset &clear : resources.resets) {
    if (!clear.shader) {
      continue;
    }
    if (!AcquireVulkanCollectiveDescriptorSet(
            adapter, *resources.reset_pipeline, 1u, clear.descriptor)) {
      return rund::AccelCheck{false, VulkanLastError(&adapter)};
    }
    std::array<VulkanStorageBinding, 1u> bindings{};
    if (!ResetBinding(adapter, clear, bindings[0], clear.binding_offset) ||
        !reset::WordAddressable(clear.range, clear.binding_offset,
                                std::numeric_limits<std::uint32_t>::max())) {
      return rund::AccelCheck{false, "accel_kernel_reset_invalid"};
    }
    if (!WriteVulkanStorageDescriptorSet(adapter, clear.descriptor, bindings)) {
      return rund::AccelCheck{false, VulkanLastError(&adapter)};
    }
  }
  return rund::AccelCheck{true, "ok"};
}

[[nodiscard]] rund::AccelCheck PrepareVulkanResets(
    VulkanAdapter &adapter, const BoundStep *const steps,
    const BoundResets *const resets, const KernelPreparationMode mode,
    std::uint32_t *const failed_node, VulkanKernelResources &resources) {
  if (resets == nullptr) {
    return rund::AccelCheck{true, "ok"};
  }
  if (resets->size() > std::numeric_limits<std::size_t>::max()) {
    RecordNode(failed_node, steps[0]);
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  resources.resets.reserve(static_cast<std::size_t>(resets->size()));
  VulkanResidentState &resident = VulkanResidents(adapter);
  for (std::uint64_t index = 0u; index < resets->size(); ++index) {
    const BoundReset &route = (*resets)[static_cast<std::size_t>(index)];
    const rund::kernel::ResidentBufferRef ref = route.ref();
    const VulkanViewTransfer *replacement = nullptr;
    if (route.external && !reset::Find(resources, route.binding, replacement)) {
      return rund::AccelCheck{false, "accel_kernel_reset_invalid"};
    }
    VulkanResidentBufferResult resolved =
        replacement == nullptr
            ? ResolveVulkanResidentBuffer(resident, ref, route.handle(),
                                          "compute_resident_id_invalid")
            : replacement->dense;
    if (!resolved.check.ok || resolved.device_buffer == nullptr) {
      return resolved.check.ok
                 ? rund::AccelCheck{false, "accel_kernel_reset_invalid"}
                 : resolved.check;
    }
    const reset::Replacement dense{
        .count = replacement == nullptr ? 0u : replacement->count,
        .element = replacement == nullptr ? 0u : replacement->element_bytes,
    };
    reset::Range range = route.range();
    if (replacement != nullptr) {
      const reset::Range proved = reset::Prove(reset::Project(ref, &dense),
                                               resolved.device_buffer->bytes);
      if (!proved.valid()) {
        return rund::AccelCheck{false, "accel_kernel_reset_invalid"};
      }
      range = proved;
    }
    const std::uint64_t reset_window =
        static_cast<std::uint64_t>(adapter.max_dispatch_groups) * 256u;
    const VulkanResetExecution execution =
        PlanVulkanResetExecution(range, mode, reset_window);
    if (!execution.ok) {
      return rund::AccelCheck{false, "accel_kernel_reset_invalid"};
    }
    resources.resets.push_back(VulkanReset{
        .resident = std::move(resolved),
        .range = range,
        .shader = execution.shader,
    });
    resources.reset_count = ::rund::detail::counter::SaturatingAdd(
        resources.reset_count, execution.commands);
    resources.reset_bytes = ::rund::detail::counter::SaturatingAdd(
        resources.reset_bytes, reset::Payload(range));
  }
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
