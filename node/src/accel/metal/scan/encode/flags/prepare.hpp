#pragma once

#include <accel/check.hpp>

#include "../../../../kernel/preparation.hpp"
#include "state.hpp"

#include <cstddef>
#include <cstring>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck PrepareMetalFlagScanEncodeState(
    MetalAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan, void *const flags_buffer,
    const std::uint64_t flags_offset, void *const output_buffer,
    const std::uint64_t output_offset, void *const totals_buffer,
    const std::uint64_t totals_offset, void *const status_buffer,
    void *const command_encoder,
    MetalFlagScanEncodeState &state) {
  if (adapter.device == nullptr || flags_buffer == nullptr ||
      output_buffer == nullptr || totals_buffer == nullptr ||
      status_buffer == nullptr || command_encoder == nullptr) {
    SetMetalLastError(adapter, "accel_metal_unavailable");
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  if (!ScanShapeOk(desc, plan) ||
      plan.element != rund::kernel::ScanElement::U32) {
    SetMetalLastError(adapter, "compute_scan_invalid");
    return rund::AccelCheck{false, "compute_scan_invalid"};
  }
  state.status = (__bridge id<MTLBuffer>)status_buffer;
  const bool pipeline_private =
      IsPipelinePrivatePreparation(CurrentKernelPreparationMode());
  if (state.status == nil ||
      (!pipeline_private && [state.status contents] == nullptr)) {
    SetMetalLastError(adapter, "accel_metal_buffer_unavailable");
    return rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
  }
  if (!pipeline_private) {
    std::memset([state.status contents], 0, sizeof(rund::kernel::u32));
  }
  if (!CompileMetalScanFlagPipelines(adapter, state.block_handle,
                                     state.prefix_handle,
                                     state.offset_handle)) {
    SetMetalLastError(adapter, "accel_metal_pipeline_unavailable");
    return rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
  }
  state.block = (__bridge id<MTLComputePipelineState>)state.block_handle.get();
  state.prefix =
      (__bridge id<MTLComputePipelineState>)state.prefix_handle.get();
  state.offset =
      (__bridge id<MTLComputePipelineState>)state.offset_handle.get();
  state.encoder = (__bridge id<MTLComputeCommandEncoder>)command_encoder;
  state.flags = (__bridge id<MTLBuffer>)flags_buffer;
  state.output = (__bridge id<MTLBuffer>)output_buffer;
  state.flags_offset = static_cast<NSUInteger>(flags_offset);
  state.output_offset = static_cast<NSUInteger>(output_offset);
  state.totals_offset = static_cast<NSUInteger>(totals_offset);
  state.totals = (__bridge id<MTLBuffer>)totals_buffer;
  state.element_count = plan.element_count;
  state.block_size = plan.block_size;
  state.block_count = plan.block_count;
  state.block_threads = static_cast<NSUInteger>(kMetalScanWidth);
  state.prefix_threads = static_cast<NSUInteger>(kMetalScanWidth);
  if (state.block == nil || state.prefix == nil || state.offset == nil ||
      state.encoder == nil) {
    SetMetalLastError(adapter, "accel_metal_command_unavailable");
    return rund::AccelCheck{false, "accel_metal_command_unavailable"};
  }
  if (state.block_size > kMetalScanMaxBlockSize || state.block_threads == 0u ||
      state.block_threads > [state.block maxTotalThreadsPerThreadgroup] ||
      (plan.pass_count == 2u &&
       (state.block_threads > [state.offset maxTotalThreadsPerThreadgroup] ||
        state.prefix_threads > [state.prefix maxTotalThreadsPerThreadgroup]))) {
    SetMetalLastError(adapter, "compute_scan_invalid");
    return rund::AccelCheck{false, "compute_scan_invalid"};
  }
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
