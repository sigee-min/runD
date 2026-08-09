#pragma once

#include <accel/check.hpp>

#include "state.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck
LoadMetalRangeState(MetalAdapter &adapter,
                    const std::shared_ptr<void> &resources,
                    void *const command_encoder, MetalRangeState &state) {
  state.range = static_cast<MetalRangeResources *>(resources.get());
  if (state.range == nullptr || state.range->adapter != &adapter) {
    SetMetalLastError(adapter, "compute_range_aggregate_invalid");
    return rund::AccelCheck{false, "compute_range_aggregate_invalid"};
  }
  if (state.range->stage_count == 0u ||
      state.range->stage_count != state.range->range.stage_count()) {
    SetMetalLastError(adapter, "compute_range_aggregate_invalid");
    return rund::AccelCheck{false, "compute_range_aggregate_invalid"};
  }
  const std::optional<RangeExec> execution =
      RangeExec::from(state.range->range);
  if (!execution.has_value()) {
    SetMetalLastError(adapter, "compute_range_aggregate_invalid");
    return rund::AccelCheck{false, "compute_range_aggregate_invalid"};
  }
  for (std::size_t index = 0u; index < state.range->stage_count; ++index) {
    if (state.range->pipelines[index] == nullptr ||
        !execution->stage_dispatch_fits(
            index, std::numeric_limits<std::uint32_t>::max())) {
      SetMetalLastError(adapter, "compute_dispatch_overflow");
      return rund::AccelCheck{false, "compute_dispatch_overflow"};
    }
  }
  state.encoder = (__bridge id<MTLComputeCommandEncoder>)command_encoder;
  state.input = (__bridge id<MTLBuffer>)state.range->input.device_buffer.get();
  state.output =
      (__bridge id<MTLBuffer>)state.range->output.device_buffer.get();
  if (state.encoder == nil || state.input == nil || state.output == nil) {
    SetMetalLastError(adapter, "accel_metal_command_unavailable");
    return rund::AccelCheck{false, "accel_metal_command_unavailable"};
  }
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
