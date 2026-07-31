#pragma once

#include <accel/check.hpp>

#include "shape.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] rund::AccelCheck PrepareMetalScanEncodeState(
    MetalAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan, void *const input_buffer,
    void *const output_buffer, void *const totals_buffer,
    void *const status_buffer, void *const command_encoder,
    MetalScanEncodeState &state, const std::shared_ptr<void> *const block,
    const std::shared_ptr<void> *const prefix,
    const std::shared_ptr<void> *const offset) {
  const rund::AccelCheck inputs = CheckMetalScanEncodeInputs(
      adapter, desc, plan, input_buffer, output_buffer, totals_buffer,
      status_buffer, command_encoder);
  if (!inputs.ok) {
    return inputs;
  }

  const rund::AccelCheck status =
      ResetMetalScanStatusBuffer(adapter, status_buffer);
  if (!status.ok) {
    return status;
  }

  rund::AccelCheck pipelines{false, "accel_metal_pipeline_unavailable"};
  if (block != nullptr && prefix != nullptr && offset != nullptr) {
    state.block_handle = *block;
    state.prefix_handle = *prefix;
    state.offset_handle = *offset;
    state.block =
        (__bridge id<MTLComputePipelineState>)state.block_handle.get();
    state.prefix =
        (__bridge id<MTLComputePipelineState>)state.prefix_handle.get();
    state.offset =
        (__bridge id<MTLComputePipelineState>)state.offset_handle.get();
    pipelines =
        state.block != nil && state.prefix != nil && state.offset != nil
            ? rund::AccelCheck{true, "ok"}
            : rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
  } else {
    pipelines = LoadMetalScanPipelines(adapter, plan, state);
  }
  if (!pipelines.ok) {
    return pipelines;
  }

  state.encoder = (__bridge id<MTLComputeCommandEncoder>)command_encoder;
  if (state.encoder == nil) {
    SetMetalLastError(adapter, "accel_metal_command_unavailable");
    return rund::AccelCheck{false, "accel_metal_command_unavailable"};
  }

  BindMetalScanPlanShape(plan, state);
  return CheckMetalScanThreadShape(adapter, plan, state);
}
#endif

} // namespace rund::node::accel::detail
