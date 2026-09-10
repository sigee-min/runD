#include "local.hpp"

#include "src/accel/backend/token.hpp"
#include "src/accel/metal/state.hpp"
#include <node/accel/buffer.hpp>
#include <node/accel/pick.hpp>

#include <memory>
#include <mutex>

namespace node_accel_contract::backend_runtime {
namespace {

namespace detail = rund::node::accel::detail;

} // namespace

bool CheckMetalCounters(const rund::AccelDevice &pick) {
  if (!pick.check.ok) {
    return node_accel_contract::MetalFailsClosed(pick);
  }
  const std::shared_ptr<detail::PickToken> token = detail::AdmitPick(pick);
  const rund::AccelDevice *const raw = token == nullptr ? nullptr : &token->raw;
  auto *const adapter =
      raw == nullptr
          ? nullptr
          : static_cast<detail::MetalAdapter *>(raw->backend.context);
  if (adapter == nullptr) {
    return false;
  }
  {
    std::lock_guard lock{adapter->mutex};
    detail::MetalRuntimeStats &stats = adapter->stats;
    OverflowCounter(stats.runtime.run.work.dispatch_count);
    OverflowCounter(stats.runtime.run.work.command_submit_count);
    OverflowCounter(stats.runtime.run.allocations.pipeline_compile_count);
    OverflowCounter(stats.runtime.run.allocations.pipeline_cache_hit_count);
    OverflowCounter(stats.runtime.run.allocations.buffer_allocation_count);
    OverflowCounter(stats.runtime.run.allocations.buffer_reuse_hit_count);
    OverflowCounter(stats.runtime.run.transfer.host_to_device_bytes);
    OverflowCounter(stats.runtime.run.transfer.device_to_host_bytes);
    OverflowCounter(stats.runtime.run.time.accel_kernel_ns);
    OverflowCounter(stats.runtime.run.time.accel_timestamp_count);
    OverflowCounter(stats.runtime.run.time.shader_compile_ns);
    OverflowCounter(stats.runtime.run.time.spirv_compile_ns);
    OverflowCounter(stats.runtime.run.time.pipeline_create_ns);
    OverflowCounter(stats.runtime.run.time.descriptor_setup_ns);
    OverflowCounter(stats.runtime.run.time.command_submit_wait_ns);
    OverflowCounter(stats.runtime.run.time.readback_ns);
  }
  const rund::RuntimeStats saturated =
      rund::node::accel::ReadRuntimeStats(pick);
  const bool saturated_ok =
      saturated.outcome.ok &&
      CountersEqual(kCounterMaximum,
                    {saturated.run.work.dispatch_count,
                     saturated.run.work.command_submit_count,
                     saturated.run.allocations.pipeline_compile_count,
                     saturated.run.allocations.pipeline_cache_hit_count,
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
                     saturated.run.time.readback_ns});
  rund::node::accel::ResetRuntimeStats(pick);
  const rund::RuntimeStats reset = rund::node::accel::ReadRuntimeStats(pick);
  return saturated_ok && reset.outcome.ok &&
         CountersEqual(0u, {reset.run.work.dispatch_count,
                            reset.run.work.command_submit_count,
                            reset.run.allocations.pipeline_compile_count,
                            reset.run.allocations.pipeline_cache_hit_count,
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
}

} // namespace node_accel_contract::backend_runtime
