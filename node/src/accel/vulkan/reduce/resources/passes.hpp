#pragma once

#include "../local.hpp"

#include <kernel/program/compute/model.hpp>
#include <kernel/program/compute/reduce/plan.hpp>

#include <cstddef>
#include <cstring>
#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

inline constexpr std::size_t kMaxReducePasses = 64u;

[[nodiscard]] bool
StoreVulkanReduceParams(VulkanReduceEncodeResources &resources,
                        const std::size_t index,
                        const ReducePassParams &value) {
  if (resources.params.mapped == nullptr ||
      index >= resources.descriptor_sets.size()) {
    return false;
  }
  auto *const bytes = static_cast<std::byte *>(resources.params.mapped);
  std::memcpy(bytes + index * resources.params_stride, &value, sizeof(value));
  return true;
}

[[nodiscard]] bool
CreateVulkanReducePassResources(VulkanAdapter &adapter,
                                VulkanReduceEncodeResources &resources) {
  if (resources.plan.pass_count >
      static_cast<rund::kernel::u64>(std::numeric_limits<std::size_t>::max())) {
    SetVulkanLastError(adapter, "compute_reduce_invalid");
    return false;
  }
  const auto count = static_cast<std::size_t>(resources.plan.pass_count);
  const VkDeviceSize alignment = adapter.storage_align;
  constexpr VkDeviceSize value_bytes = sizeof(ReducePassParams);
  if (count == 0u || count > kMaxReducePasses || alignment == 0u ||
      value_bytes >
          std::numeric_limits<VkDeviceSize>::max() - (alignment - 1u)) {
    SetVulkanLastError(adapter, "compute_reduce_invalid");
    return false;
  }
  resources.params_stride =
      ((value_bytes + alignment - 1u) / alignment) * alignment;
  if (resources.params_stride < value_bytes ||
      static_cast<VkDeviceSize>(count) >
          std::numeric_limits<VkDeviceSize>::max() / resources.params_stride) {
    SetVulkanLastError(adapter, "compute_reduce_invalid");
    return false;
  }
  const VkDeviceSize params_bytes =
      static_cast<VkDeviceSize>(count) * resources.params_stride;
  resources.descriptor_sets.resize(count, VK_NULL_HANDLE);
  if (!CreateVulkanBuffer(adapter, params_bytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          resources.params) ||
      !ReserveVulkanCollectiveDescriptorSets(adapter, *resources.pipeline,
                                             kReduceDescriptorCount,
                                             resources.plan.pass_count)) {
    return false;
  }

  if (rund::kernel::ReduceUsesWidePartials(resources.plan.op)) {
    const rund::kernel::u64 groups = resources.plan.first_pass_group_count;
    const ReducePassParams first{
        0u,
        0u,
        resources.plan.element_count,
        groups,
        resources.plan.pass_count == 1u ? 1u : 0u,
        1u,
        static_cast<rund::kernel::u32>(
            rund::kernel::ComputeCountBytes(resources.plan.count_source) /
            sizeof(rund::kernel::u32))};
    if (!StoreVulkanReduceParams(resources, 0u, first) ||
        !CreateVulkanReducePassDescriptorSet(adapter, resources, 0u,
                                             resources.input)) {
      return false;
    }
    if (resources.plan.pass_count == 2u) {
      const ReducePassParams finish{0u, 0u, groups, 1u, 1u, 0u, 0u};
      if (!StoreVulkanReduceParams(resources, 1u, finish) ||
          !CreateVulkanReducePassDescriptorSet(adapter, resources, 1u,
                                               resources.input)) {
        return false;
      }
    }
    return true;
  }

  rund::kernel::u64 current = resources.plan.element_count;
  rund::kernel::u64 read_offset = 0u;
  rund::kernel::u64 write_offset = 0u;
  bool done = false;
  for (rund::kernel::u64 pass = 0u; pass < resources.plan.pass_count; ++pass) {
    const rund::kernel::u64 next =
        rund::kernel::ReduceGroupCount(current, resources.plan.block_size);
    const bool final_pass = next == 1u;
    const ReducePassParams value{
        read_offset,
        write_offset,
        current,
        next,
        final_pass ? 1u : 0u,
        pass == 0u ? 1u : 0u,
        static_cast<rund::kernel::u32>(
            pass == 0u
                ? rund::kernel::ComputeCountBytes(resources.plan.count_source) /
                      sizeof(rund::kernel::u32)
                : 0u)};
    const auto index = static_cast<std::size_t>(pass);
    const VulkanStorageBinding read_buffer =
        pass == 0u ? resources.input
                   : VulkanStorageBindingFor(resources.partial);
    if (!StoreVulkanReduceParams(resources, index, value) ||
        !CreateVulkanReducePassDescriptorSet(adapter, resources, index,
                                             read_buffer)) {
      return false;
    }
    if (final_pass) {
      done = index + 1u == count;
      break;
    }
    read_offset = write_offset;
    write_offset += next;
    current = next;
  }
  if (!done) {
    SetVulkanLastError(adapter, "compute_reduce_invalid");
    return false;
  }
  return true;
}

} // namespace
#endif

} // namespace rund::node::accel::detail
