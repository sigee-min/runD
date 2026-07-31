#include <accel/device.hpp>
#include <accel/runtime.hpp>

#include "local.hpp"
#include "../command.hpp"

#include <mutex>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
rund::RuntimeStats ReadVulkanRuntimeStats(const rund::AccelDevice &pick) {
  if (!VulkanPickOwnsAdapter(pick)) {
    return rund::RuntimeStats{.ok = false,
                              .reason = "accel_buffer_backend_unavailable"};
  }
  auto *const adapter = static_cast<VulkanAdapter *>(pick.backend.context);
  std::unique_lock<std::mutex> lock{adapter->mutex};
  adapter->host_readback_cv.wait(
      lock, [adapter] { return adapter->active_host_readbacks == 0u; });
  return rund::RuntimeStats{
      .dispatch_count = adapter->dispatch_count,
      .command_submit_count = adapter->command_submit_count,
      .command_capacity = kVulkanCommandCapacity,
      .command_inflight_peak = adapter->command_inflight_peak,
      .command_capacity_rejection_count =
          adapter->command_capacity_rejection_count,
      .pipeline_compile_count = adapter->pipeline_compile_count,
      .pipeline_cache_hit_count = adapter->pipeline_cache_hit_count,
      .descriptor_pool_create_count = adapter->descriptor_pool_create_count,
      .descriptor_set_allocate_count = adapter->descriptor_set_allocate_count,
      .descriptor_reuse_hit_count = adapter->descriptor_reuse_hit_count,
      .buffer_allocation_count = adapter->buffer_allocation_count,
      .buffer_reuse_hit_count = adapter->buffer_reuse_hit_count,
      .host_to_device_bytes = adapter->host_to_device_bytes,
      .device_to_host_bytes = adapter->device_to_host_bytes,
      .accel_kernel_ns = adapter->accel_kernel_ns,
      .accel_timestamp_count = adapter->accel_timestamp_count,
      .accel_timestamp_source = adapter->accel_timestamp_source,
      .shader_compile_ns = adapter->shader_compile_ns,
      .spirv_compile_ns = adapter->spirv_compile_ns,
      .pipeline_create_ns = adapter->pipeline_create_ns,
      .descriptor_setup_ns = adapter->descriptor_setup_ns,
      .command_submit_wait_ns = adapter->command_submit_wait_ns,
      .readback_ns = adapter->readback_ns,
      .ok = true,
      .reason = "ok",
  };
}

void ResetVulkanRuntimeStats(const rund::AccelDevice &pick) {
  if (!VulkanPickOwnsAdapter(pick)) {
    return;
  }
  auto *const adapter = static_cast<VulkanAdapter *>(pick.backend.context);
  std::unique_lock<std::mutex> lock{adapter->mutex};
  adapter->host_readback_cv.wait(
      lock, [adapter] { return adapter->active_host_readbacks == 0u; });
  // A reset is an evidence epoch boundary. Waiting makes all prior transfer
  // and compute submissions belong to the old epoch instead of racing their
  // completion counters into the next one.
  WaitForVulkanCommands(*adapter, lock);
  adapter->dispatch_count = 0u;
  adapter->command_submit_count = 0u;
  adapter->command_inflight_peak = 0u;
  adapter->command_capacity_rejection_count = 0u;
  adapter->pipeline_compile_count = 0u;
  adapter->pipeline_cache_hit_count = 0u;
  adapter->descriptor_pool_create_count = 0u;
  adapter->descriptor_set_allocate_count = 0u;
  adapter->descriptor_reuse_hit_count = 0u;
  adapter->buffer_allocation_count = 0u;
  adapter->buffer_reuse_hit_count = 0u;
  adapter->host_to_device_bytes = 0u;
  adapter->device_to_host_bytes = 0u;
  adapter->accel_kernel_ns = 0u;
  adapter->accel_timestamp_count = 0u;
  adapter->accel_timestamp_source = adapter->timestamp_query_available
                                        ? "vulkan_timestamp_query"
                                        : "unavailable";
  adapter->shader_compile_ns = 0u;
  adapter->spirv_compile_ns = 0u;
  adapter->pipeline_create_ns = 0u;
  adapter->descriptor_setup_ns = 0u;
  adapter->command_submit_wait_ns = 0u;
  adapter->readback_ns = 0u;
}
#endif

} // namespace rund::node::accel::detail
