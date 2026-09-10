#include "local.hpp"

#include "src/accel/backend/token.hpp"
#include "src/accel/vulkan/adapter/access.hpp"
#include <node/accel/buffer.hpp>
#include <node/accel/pick.hpp>

#include <memory>
#include <mutex>

namespace node_accel_contract::backend_runtime {
namespace {

namespace detail = rund::node::accel::detail;

} // namespace

bool CheckVulkanCounters(const rund::AccelDevice &pick) {
  if (!pick.check.ok) {
    return node_accel_contract::vulkan::FailureReasonIsPrecise(pick);
  }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const std::shared_ptr<detail::PickToken> token = detail::AdmitPick(pick);
  const rund::AccelDevice *const raw = token == nullptr ? nullptr : &token->raw;
  detail::VulkanAdapter *const adapter =
      raw == nullptr ? nullptr : detail::CheckedVulkanAdapter(*raw);
  if (adapter == nullptr) {
    return false;
  }
  {
    std::lock_guard lock{adapter->mutex};
    OverflowCounter(adapter->dispatch_count);
    OverflowCounter(adapter->command_submit_count);
    OverflowCounter(adapter->command_capacity_rejection_count);
    OverflowCounter(adapter->pipeline_compile_count);
    OverflowCounter(adapter->pipeline_cache_hit_count);
    OverflowCounter(adapter->descriptor_pool_create_count);
    OverflowCounter(adapter->descriptor_set_allocate_count);
    OverflowCounter(adapter->descriptor_reuse_hit_count);
    OverflowCounter(adapter->buffer_allocation_count);
    OverflowCounter(adapter->buffer_reuse_hit_count);
    OverflowCounter(adapter->host_to_device_bytes);
    OverflowCounter(adapter->device_to_host_bytes);
    OverflowCounter(adapter->accel_kernel_ns);
    OverflowCounter(adapter->accel_timestamp_count);
    OverflowCounter(adapter->shader_compile_ns);
    OverflowCounter(adapter->spirv_compile_ns);
    OverflowCounter(adapter->pipeline_create_ns);
    OverflowCounter(adapter->descriptor_setup_ns);
    OverflowCounter(adapter->command_submit_wait_ns);
    OverflowCounter(adapter->readback_ns);
    adapter->command_inflight_peak = detail::kVulkanCommandCapacity;
  }
  const rund::RuntimeStats saturated =
      rund::node::accel::ReadRuntimeStats(pick);
  const bool saturated_ok =
      saturated.outcome.ok &&
      CountersEqual(kCounterMaximum,
                    {saturated.run.work.dispatch_count,
                     saturated.run.work.command_submit_count,
                     saturated.run.work.command_capacity_rejection_count,
                     saturated.run.allocations.pipeline_compile_count,
                     saturated.run.allocations.pipeline_cache_hit_count,
                     saturated.run.allocations.descriptor_pool_create_count,
                     saturated.run.allocations.descriptor_set_allocate_count,
                     saturated.run.allocations.descriptor_reuse_hit_count,
                     saturated.run.allocations.buffer_allocation_count,
                     saturated.run.allocations.buffer_reuse_hit_count,
                     saturated.run.transfer.host_to_device_bytes,
                     saturated.run.transfer.device_to_host_bytes,
                     saturated.run.time.accel_kernel_ns,
                     saturated.run.time.accel_timestamp_count,
                     saturated.run.time.shader_compile_ns,
                     saturated.run.time.spirv_compile_ns,
                     saturated.run.time.pipeline_create_ns,
                     saturated.run.time.descriptor_setup_ns,
                     saturated.run.time.command_submit_wait_ns,
                     saturated.run.time.readback_ns}) &&
      saturated.run.work.command_capacity == detail::kVulkanCommandCapacity &&
      saturated.run.work.command_inflight_peak ==
          detail::kVulkanCommandCapacity;
  rund::node::accel::ResetRuntimeStats(pick);
  const rund::RuntimeStats reset = rund::node::accel::ReadRuntimeStats(pick);
  return saturated_ok && reset.outcome.ok &&
         reset.run.work.command_capacity == detail::kVulkanCommandCapacity &&
         reset.run.work.command_inflight_peak == 0u &&
         reset.run.work.command_capacity_rejection_count == 0u &&
         CountersEqual(0u, {reset.run.work.dispatch_count,
                            reset.run.work.command_submit_count,
                            reset.run.allocations.pipeline_compile_count,
                            reset.run.allocations.pipeline_cache_hit_count,
                            reset.run.allocations.descriptor_pool_create_count,
                            reset.run.allocations.descriptor_set_allocate_count,
                            reset.run.allocations.descriptor_reuse_hit_count,
                            reset.run.allocations.buffer_allocation_count,
                            reset.run.allocations.buffer_reuse_hit_count,
                            reset.run.transfer.host_to_device_bytes,
                            reset.run.transfer.device_to_host_bytes,
                            reset.run.time.accel_kernel_ns,
                            reset.run.time.accel_timestamp_count,
                            reset.run.time.shader_compile_ns,
                            reset.run.time.spirv_compile_ns,
                            reset.run.time.pipeline_create_ns,
                            reset.run.time.descriptor_setup_ns,
                            reset.run.time.command_submit_wait_ns,
                            reset.run.time.readback_ns});
#else
  return false;
#endif
}

} // namespace node_accel_contract::backend_runtime
