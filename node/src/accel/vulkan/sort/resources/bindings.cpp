#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../buffer/resident/batch.hpp"
#include "../local/api.hpp"
namespace rund::node::accel::detail {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
rund::AccelCheck LookupVulkanSortResidentBuffers(
    VulkanAdapter &adapter, const rund::AccelDevice &pick,
    const rund::kernel::SortPlan &plan, const SortBinds &bindings,
    VulkanSortResidentBuffers &buffers) {
  VulkanResidentBufferResult read_keys{}, read_values{};
  VulkanResidentBufferResult write_keys{};
  VulkanResidentBufferResult write_values{};
  VulkanResidentBufferResult logical_count{};
  buffers.identity_values = plan.value == rund::kernel::SortValue::IdentityU32;
  VulkanResidentReq reqs[5] = {
      {bindings.read_keys, bindings.read_keys_handle, &read_keys},
      {bindings.write_keys, bindings.write_keys_handle, &write_keys},
      {bindings.write_values, bindings.write_values_handle, &write_values},
      {},
      {}};
  std::size_t count = 3u;
  if (!buffers.identity_values) {
    reqs[1] = {bindings.read_values, bindings.read_values_handle,
               &read_values};
    reqs[2] = {bindings.write_keys, bindings.write_keys_handle, &write_keys};
    reqs[3] = {bindings.write_values, bindings.write_values_handle,
               &write_values};
    count = 4u;
  }
  const bool bounded =
      plan.count_source != rund::kernel::ComputeCountSource::Descriptor;
  if (bounded) {
    reqs[count++] = {bindings.logical_count, bindings.logical_count_handle,
                     &logical_count};
  }
  LookupVulkanResidentBatch(pick, reqs, count, "compute_resident_id_invalid");
  if (!bounded) {
    logical_count = read_keys;
  }
  if (!read_keys.check.ok ||
      (!buffers.identity_values && !read_values.check.ok) ||
      !write_keys.check.ok || !write_values.check.ok ||
      read_keys.device_buffer == nullptr ||
      (!buffers.identity_values && read_values.device_buffer == nullptr) ||
      write_keys.device_buffer == nullptr ||
      write_values.device_buffer == nullptr || !logical_count.check.ok ||
      logical_count.device_buffer == nullptr) {
    const char *reason = "accel_buffer_unavailable";
    if (!read_keys.check.ok) {
      reason = read_keys.check.reason;
    } else if (!buffers.identity_values && !read_values.check.ok) {
      reason = read_values.check.reason;
    } else if (!write_keys.check.ok) {
      reason = write_keys.check.reason;
    } else if (!write_values.check.ok) {
      reason = write_values.check.reason;
    } else if (!logical_count.check.ok) {
      reason = logical_count.check.reason;
    }
    SetVulkanLastError(adapter, reason);
    return rund::AccelCheck{false, reason};
  }
  const auto requested = [](const VulkanResidentBufferResult &buffer) {
    return VulkanStorageBinding{
        buffer.device_buffer, static_cast<VkDeviceSize>(buffer.ref.offset_bytes),
        static_cast<VkDeviceSize>(buffer.ref.count * buffer.ref.element_bytes)};
  };
  buffers.read_keys = requested(read_keys);
  if (!buffers.identity_values) {
    buffers.read_values = requested(read_values);
  }
  buffers.write_keys = requested(write_keys);
  buffers.write_values = requested(write_values);
  buffers.logical_count = requested(logical_count);
  return rund::AccelCheck{true, "ok"};
}
#endif
} // namespace rund::node::accel::detail
