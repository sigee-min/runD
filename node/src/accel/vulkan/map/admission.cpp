#include "../buffer/create.hpp"
#include "../buffer/resident/lookup.hpp"

#include "admission.hpp"

#include "../buffer/create/telemetry.hpp"
#include "../collective/pipeline.hpp"
#include "../descriptor.hpp"
#include "../../kernel/backend/run.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

bool PrepareVulkanMapControl(
    const rund::AccelDevice &pick, const BoundControl &bound,
    const std::vector<rund::kernel::ComputeDispatchWindow> &windows,
    VulkanMapEncodeResources &resources) {
  if (resources.prepared == nullptr ||
      (!bound.active() && resources.prepared->checks.empty())) {
    return true;
  }
  if (windows.empty() ||
      windows.size() > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  resources.mode = VulkanMapMode::Direct;
  resources.control = bound.control;
  if (bound.control.has_count()) {
    if (bound.count == nullptr || bound.count_handle == nullptr) {
      return false;
    }
    resources.control_count =
        LookupVulkanResidentBuffer(pick, *bound.count, *bound.count_handle);
    if (!resources.control_count.check.ok) {
      return false;
    }
    // Lookup returns the canonical allocation identity. Control addressing,
    // however, is relative to the bound view and must retain its suballocation
    // offset just like ordinary Map bindings do.
    resources.control_count.ref = *bound.count;
  }
  if (bound.control.has_predicate()) {
    if (bound.predicate == nullptr || bound.predicate_handle == nullptr) {
      return false;
    }
    resources.control_predicate = LookupVulkanResidentBuffer(
        pick, *bound.predicate, *bound.predicate_handle);
    if (!resources.control_predicate.check.ok) {
      return false;
    }
    resources.control_predicate.ref = *bound.predicate;
  }
  const VkDeviceSize args_bytes = windows.size() * 4u * sizeof(std::uint32_t);
  VulkanBuffer args{};
  if (!CreateVulkanBuffer(*resources.adapter, args_bytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          args, nullptr, VulkanMemoryUse::Device)) {
    return false;
  }
  resources.control_args = ScopedBuffer{*resources.adapter, args, args_bytes};
  resources.control_pipeline = resources.prepared->control_pipeline;
  if (resources.control_pipeline == nullptr ||
      !AcquireVulkanCollectiveDescriptorSet(*resources.adapter,
                                            *resources.control_pipeline, 4u,
                                            resources.control_descriptor) ||
      !CreateVulkanStatus(*resources.adapter, 2u * sizeof(std::uint32_t),
                          resources.control_status)) {
    return false;
  }
  const VulkanBuffer *const count = bound.control.has_count()
                                        ? resources.control_count.device_buffer
                                        : &resources.control_args.buffer;
  const VulkanBuffer *const predicate =
      bound.control.has_predicate() ? resources.control_predicate.device_buffer
                                    : &resources.control_args.buffer;
  if (count == nullptr || predicate == nullptr) {
    return false;
  }
  const auto control_binding =
      [&](const VulkanResidentBufferResult &resident,
          const std::uint64_t offset,
          const rund::kernel::GraphControlSource source, std::uint64_t &base,
          VulkanStorageBinding &binding) {
        const std::uint64_t width =
            source == rund::kernel::GraphControlSource::U64 ? 8u : 4u;
        const std::uint64_t alignment = resources.adapter->storage_align;
        if (resident.device_buffer == nullptr || alignment == 0u ||
            offset > std::numeric_limits<std::uint64_t>::max() -
                         resident.ref.offset_bytes ||
            resident.ref.offset_bytes + offset >
                std::numeric_limits<std::uint64_t>::max() - width) {
          return false;
        }
        const std::uint64_t absolute = resident.ref.offset_bytes + offset;
        base = absolute - absolute % alignment;
        const std::uint64_t range = absolute + width - base;
        if (base > resident.device_buffer->bytes ||
            range > resident.device_buffer->bytes - base ||
            range > resources.adapter->storage_limit) {
          return false;
        }
        binding = VulkanStorageBinding{resident.device_buffer, base, range};
        return true;
      };
  VulkanStorageBinding count_binding{count, 0u, args_bytes};
  VulkanStorageBinding predicate_binding{predicate, 0u, args_bytes};
  if ((bound.control.has_count() &&
       !control_binding(
           resources.control_count, bound.control.count_byte_offset,
           bound.control.count_source, resources.count_base, count_binding)) ||
      (bound.control.has_predicate() &&
       !control_binding(resources.control_predicate,
                        bound.control.predicate_byte_offset,
                        bound.control.predicate_source,
                        resources.predicate_base, predicate_binding))) {
    return false;
  }
  const std::array<VulkanStorageBinding, 4u> bindings{
      count_binding,
      predicate_binding,
      VulkanStorageBinding{&resources.control_args.buffer, 0u, args_bytes},
      VulkanStorageBinding{&resources.control_status.device, 0u,
                           2u * sizeof(std::uint32_t)},
  };
  if (!WriteVulkanStorageDescriptorSet(
          *resources.adapter, resources.control_descriptor, bindings)) {
    return false;
  }
  if (resources.prepared->checks.empty()) {
    return true;
  }
  const std::uint64_t capacity =
      windows.back().begin_sequence + windows.back().tile_count;
  std::vector<VulkanStorageBinding> check_bindings;
  check_bindings.reserve(resources.prepared->checks.size() + 3u);
  check_bindings.push_back(count_binding);
  check_bindings.push_back(predicate_binding);
  resources.check_bases.clear();
  resources.check_bases.reserve(resources.prepared->checks.size());
  for (const VulkanMapCheck &check : resources.prepared->checks) {
    const auto *const ref =
        resources.bindings.resident_inputs.ref(check.binding);
    const VulkanResidentBufferResult &resident =
        resources.resident.input(check.binding);
    const std::uint64_t alignment = resources.adapter->storage_align;
    if (check.limit == 0u || ref == nullptr || ref->element_bytes != 4u ||
        ref->stride_bytes < 4u || (ref->stride_bytes & 3u) != 0u ||
        resident.device_buffer == nullptr || alignment == 0u ||
        capacity == 0u ||
        !rund::kernel::checked::mul(capacity - 1u, ref->stride_bytes) ||
        !rund::kernel::checked::add((capacity - 1u) * ref->stride_bytes,
                                    ref->element_bytes)) {
      return false;
    }
    const std::uint64_t base =
        ref->offset_bytes - ref->offset_bytes % alignment;
    if (ref->offset_bytes - base != check.offset ||
        ref->stride_bytes != check.stride) {
      return false;
    }
    resources.check_bases.push_back(base);
    const std::uint64_t payload =
        (capacity - 1u) * ref->stride_bytes + ref->element_bytes;
    if (check.offset > std::numeric_limits<std::uint64_t>::max() - payload) {
      return false;
    }
    const std::uint64_t range = check.offset + payload;
    if (base > resident.device_buffer->bytes ||
        range > resident.device_buffer->bytes - base ||
        range > resources.adapter->storage_limit) {
      return false;
    }
    check_bindings.push_back(
        VulkanStorageBinding{resident.device_buffer, base, range});
  }
  check_bindings.push_back(VulkanStorageBinding{
      &resources.control_status.device, 0u, 2u * sizeof(std::uint32_t)});
  resources.check_pipeline = resources.prepared->check_pipeline;
  return resources.check_pipeline != nullptr &&
         AcquireVulkanCollectiveDescriptorSet(
             *resources.adapter, *resources.check_pipeline,
             static_cast<std::uint32_t>(check_bindings.size()),
             resources.check_descriptor) &&
         WriteVulkanStorageDescriptorSet(
             *resources.adapter, resources.check_descriptor,
             check_bindings.data(),
             static_cast<std::uint32_t>(check_bindings.size()));
}

bool BindVulkanGeneratedMapAdmission(VulkanMapEncodeResources &map,
                                     const VulkanBuffer &gate,
                                     const VulkanBuffer &summary) noexcept {
  if (map.mode != VulkanMapMode::Generated || map.adapter == nullptr ||
      map.generated_control_descriptor == VK_NULL_HANDLE ||
      gate.buffer == VK_NULL_HANDLE || summary.buffer == VK_NULL_HANDLE ||
      map.control_args.buffer.buffer == VK_NULL_HANDLE ||
      map.control_status.device.buffer == VK_NULL_HANDLE) {
    return false;
  }
  const VkDeviceSize args_bytes = map.control_args.used_bytes;
  const auto source_binding = [](const VulkanResidentBufferResult &resident,
                                 const std::uint64_t base,
                                 const bool present) {
    if (!present) {
      return VulkanStorageBinding{};
    }
    if (resident.device_buffer == nullptr ||
        base >= resident.device_buffer->bytes) {
      return VulkanStorageBinding{};
    }
    return VulkanStorageBinding{resident.device_buffer, base,
                                resident.device_buffer->bytes - base};
  };
  const VulkanStorageBinding count_binding =
      map.control.has_count()
          ? source_binding(map.control_count, map.count_base, true)
          : VulkanStorageBinding{&map.control_args.buffer, 0u, args_bytes};
  const VulkanStorageBinding predicate_binding =
      map.control.has_predicate()
          ? source_binding(map.control_predicate, map.predicate_base, true)
          : VulkanStorageBinding{&map.control_args.buffer, 0u, args_bytes};
  if (count_binding.buffer == nullptr || count_binding.range == 0u ||
      predicate_binding.buffer == nullptr || predicate_binding.range == 0u) {
    return false;
  }
  const std::array<VulkanStorageBinding, 6u> bindings{
      count_binding,
      predicate_binding,
      VulkanStorageBinding{&map.control_args.buffer, 0u, args_bytes},
      VulkanStorageBinding{&map.control_status.device, 0u,
                           2u * sizeof(std::uint32_t)},
      VulkanStorageBinding{&gate, 0u, gate.bytes},
      VulkanStorageBinding{&summary, 0u, summary.bytes},
  };
  if (!WriteVulkanStorageDescriptorSet(
          *map.adapter, map.generated_control_descriptor, bindings)) {
    return false;
  }
  map.generated_gate_binding = gate;
  map.generated_summary_binding = summary;
  return true;
}

#endif

} // namespace rund::node::accel::detail
