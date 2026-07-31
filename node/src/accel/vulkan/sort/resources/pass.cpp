#include <accel/check.hpp>

#include "../local/api.hpp"

#include <kernel/program/compute/model.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
rund::AccelCheck
PrepareVulkanSortDispatch(VulkanAdapter &adapter,
                          const VulkanSortResidentBuffers &buffers,
                          VulkanSortEncodeResources &resources) {
  if (!AcquireVulkanCollectiveDescriptorSet(
          adapter, *resources.dispatch_pipeline, kSortDescriptorCount,
          resources.dispatch_descriptor)) {
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }
  const VulkanStorageBinding params_binding{&resources.params, 0u,
                                            sizeof(SortParams)};
  const VulkanStorageBinding dispatch_binding =
      VulkanStorageBindingFor(resources.dispatch_args);
  const VulkanStorageBinding status_binding =
      VulkanStorageBindingFor(resources.status.device);
  const VulkanStorageBinding logical_binding = buffers.logical_count;
  const bool ready = WriteVulkanStorageDescriptorSet(
      adapter, resources.dispatch_descriptor,
      std::array<VulkanStorageBinding, kSortDescriptorCount>{
          dispatch_binding, params_binding, params_binding, params_binding,
          status_binding, params_binding, params_binding, params_binding,
          logical_binding});
  return rund::AccelCheck{ready, ready ? "ok" : VulkanLastError(&adapter)};
}

rund::AccelCheck PrepareVulkanSortPass(VulkanAdapter &adapter,
                                       const rund::kernel::SortPlan &plan,
                                       const VulkanSortResidentBuffers &buffers,
                                       const bool signed_order,
                                       const std::size_t pass,
                                       VulkanSortEncodeResources &resources,
                                       SortParams &params_value) {
  params_value = SortParams{
      .element_count = plan.element_count,
      .block_count = resources.block_count,
      .pass_index = static_cast<rund::kernel::u32>(pass),
      .identity_values = buffers.identity_values ? 1u : 0u,
      .signed_order = signed_order ? 1u : 0u,
      .pass_count = plan.radix_pass_count,
      .count_words = static_cast<rund::kernel::u32>(
          rund::kernel::ComputeCountBytes(plan.count_source) /
          sizeof(rund::kernel::u32)),
      .max_dispatch_groups = adapter.max_dispatch_groups,
      .chunk_count = resources.chunk_count,
  };
  if (!AcquireVulkanCollectiveDescriptorSet(
          adapter, *resources.classify_pipeline, kSortDescriptorCount,
          resources.descriptors[pass].classify_set) ||
      !AcquireVulkanCollectiveDescriptorSet(
          adapter, *resources.prefix_pipeline, kSortDescriptorCount,
          resources.descriptors[pass].prefix_set) ||
      !AcquireVulkanCollectiveDescriptorSet(
          adapter, *resources.base_pipeline, kSortDescriptorCount,
          resources.descriptors[pass].base_set) ||
      !AcquireVulkanCollectiveDescriptorSet(
          adapter, *resources.scatter_pipeline, kSortDescriptorCount,
          resources.descriptors[pass].scatter_set)) {
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }

  VulkanStorageBinding source_keys = buffers.read_keys;
  VulkanStorageBinding source_values =
      buffers.identity_values ? VulkanStorageBindingFor(resources.temp_values)
                              : buffers.read_values;
  if (pass != 0u) {
    source_keys = (pass % 2u) == 1u
                      ? VulkanStorageBindingFor(resources.temp_keys)
                      : buffers.write_keys;
    source_values = (pass % 2u) == 1u
                        ? VulkanStorageBindingFor(resources.temp_values)
                        : buffers.write_values;
  }
  const VulkanStorageBinding target_keys =
      (pass % 2u) == 0u ? VulkanStorageBindingFor(resources.temp_keys)
                        : buffers.write_keys;
  const VulkanStorageBinding target_values =
      (pass % 2u) == 0u ? VulkanStorageBindingFor(resources.temp_values)
                        : buffers.write_values;
  resources.target_keys[pass] = target_keys.buffer;
  resources.target_values[pass] = target_values.buffer;
  const VkDeviceSize params_offset =
      static_cast<VkDeviceSize>(pass) * resources.params_stride;
  const VulkanStorageBinding params_binding{&resources.params, params_offset,
                                            sizeof(SortParams)};
  const VulkanStorageBinding count_binding =
      VulkanStorageBindingFor(resources.block_counts);
  const VulkanStorageBinding offset_binding =
      VulkanStorageBindingFor(resources.block_offsets);
  const VulkanStorageBinding status_binding =
      VulkanStorageBindingFor(resources.status.device);
  const VulkanStorageBinding logical_binding = buffers.logical_count;

  bool ready = WriteVulkanStorageDescriptorSet(
      adapter, resources.descriptors[pass].classify_set,
      std::array<VulkanStorageBinding, kSortDescriptorCount>{
          source_keys, count_binding, params_binding, params_binding,
          status_binding, offset_binding, count_binding, params_binding,
          logical_binding});
  ready = ready && WriteVulkanStorageDescriptorSet(
                       adapter, resources.descriptors[pass].prefix_set,
                       std::array<VulkanStorageBinding, kSortDescriptorCount>{
                           params_binding, count_binding, params_binding,
                           params_binding, status_binding, offset_binding,
                           count_binding, params_binding, logical_binding});
  ready = ready && WriteVulkanStorageDescriptorSet(
                       adapter, resources.descriptors[pass].base_set,
                       std::array<VulkanStorageBinding, kSortDescriptorCount>{
                           params_binding, count_binding, params_binding,
                           params_binding, status_binding, offset_binding,
                           count_binding, params_binding, logical_binding});
  ready = ready && WriteVulkanStorageDescriptorSet(
                       adapter, resources.descriptors[pass].scatter_set,
                       std::array<VulkanStorageBinding, kSortDescriptorCount>{
                           source_keys, source_values, target_keys,
                           target_values, status_binding, offset_binding,
                           count_binding, params_binding, logical_binding});

  return rund::AccelCheck{ready, ready ? "ok" : VulkanLastError(&adapter)};
}
#endif

} // namespace rund::node::accel::detail
