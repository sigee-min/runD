#pragma once

#include <accel/check.hpp>

#include "../../../command/run.hpp"
#include "encode.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] bool ExecuteWindows(
    MetalAdapter &adapter, const std::shared_ptr<void> &pipeline_handle,
    const rund::kernel::ComputePlan &plan,
    const rund::kernel::ComputeDispatchWindow *windows,
    rund::kernel::u64 window_count, const rund::kernel::BindingSet &bindings,
    const MetalRuntimeBuffer &param_buffer,
    const MetalResidentBindings &resident,
    const std::span<const InputWindowPlan> input_plans) {
  id<MTLComputePipelineState> pipeline =
      (__bridge id<MTLComputePipelineState>)pipeline_handle.get();
  if (pipeline == nil || param_buffer.buffer == nullptr ||
      resident.bindings != &bindings || plan.output_buffer_count == 0u ||
      resident.output(0u).device_buffer == nullptr || windows == nullptr ||
      window_count == 0u) {
    SetMetalLastError(adapter, "accel_metal_command_unavailable");
    return false;
  }
  CommandRun command{};
  if (!OpenCommand(adapter, command).ok) {
    return false;
  }
  [command.encoder setComputePipelineState:pipeline];
  [command.encoder setBuffer:(__bridge id<MTLBuffer>)param_buffer.buffer.get()
                      offset:0u
                     atIndex:0u];
  rund::AccelCheck encoded{true, "ok"};
  for (rund::kernel::u64 index = 0u; index < window_count; ++index) {
    if (!EncodeResidentWindow(adapter, command.encoder, pipeline, plan,
                              windows[index], bindings, resident,
                              input_plans)) {
      encoded = rund::AccelCheck{false, "compute_binding_mismatch"};
      break;
    }
  }
  const rund::AccelCheck submit = FinishCommand(adapter, command, encoded);
  if (!submit.ok) {
    return false;
  }
  RecordMetalDispatches(adapter, window_count);
  return true;
}
#endif

} // namespace rund::node::accel::detail
