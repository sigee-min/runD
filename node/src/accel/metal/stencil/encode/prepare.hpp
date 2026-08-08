#pragma once

#include <accel/check.hpp>

#include "../../../stencil/shape.hpp"
#include "state.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck PrepareMetalStencilCommandState(
    MetalAdapter &adapter, const std::shared_ptr<void> &resources,
    void *const command_encoder, MetalStencilCommandState &state) {
  state.stencil = static_cast<MetalStencilEncodeResources *>(resources.get());
  if (state.stencil == nullptr || state.stencil->adapter != &adapter) {
    SetMetalLastError(adapter, "compute_stencil_invalid");
    return rund::AccelCheck{false, "compute_stencil_invalid"};
  }
  if (state.stencil->stage_count == 0u ||
      state.stencil->stage_count != state.stencil->range.stage_count()) {
    SetMetalLastError(adapter, "compute_stencil_invalid");
    return rund::AccelCheck{false, "compute_stencil_invalid"};
  }
  for (std::size_t index = 0u; index < state.stencil->stage_count; ++index) {
    if (state.stencil->pipelines[index] == nullptr ||
        !RangeAggregateStageDispatchFits(
            state.stencil->range, index,
            std::numeric_limits<std::uint32_t>::max())) {
      SetMetalLastError(adapter, "compute_dispatch_overflow");
      return rund::AccelCheck{false, "compute_dispatch_overflow"};
    }
  }
  state.encoder = (__bridge id<MTLComputeCommandEncoder>)command_encoder;
  state.input =
      (__bridge id<MTLBuffer>)state.stencil->input.device_buffer.get();
  state.output =
      (__bridge id<MTLBuffer>)state.stencil->output.device_buffer.get();
  if (state.encoder == nil || state.input == nil || state.output == nil) {
    SetMetalLastError(adapter, "accel_metal_command_unavailable");
    return rund::AccelCheck{false, "accel_metal_command_unavailable"};
  }
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
