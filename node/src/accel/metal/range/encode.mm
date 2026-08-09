#include <accel/check.hpp>
#include <accel/device.hpp>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include "../../kernel/backend/run.hpp"
#include "../../kernel/preparation.hpp"
#include "../../kernel/scratch.hpp"
#include "../../range_aggregate/plan.hpp"
#include "../pipeline/template.hpp"
#include "../scratch.hpp"
#include "encode/dispatch.hpp"
#include "local.hpp"
#include "pipeline/store.hpp"
#include "resources/prepare.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace rund::node::accel::detail {

rund::AccelCheck EncodeMetalRange(MetalAdapter &adapter,
                                  const std::shared_ptr<void> &resources,
                                  void *command_encoder) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  MetalRangeState state{};
  const rund::AccelCheck prepared =
      LoadMetalRangeState(adapter, resources, command_encoder, state);
  if (!prepared.ok) {
    return prepared;
  }
  if (state.range->controlled) {
    id<MTLComputePipelineState> control =
        (__bridge id<MTLComputePipelineState>)
            state.range->control_pipeline.get();
    id<MTLBuffer> count =
        (__bridge id<MTLBuffer>)state.range->control_count.device_buffer.get();
    id<MTLBuffer> params =
        (__bridge id<MTLBuffer>)state.range->control_params.buffer.get();
    id<MTLBuffer> indirect =
        (__bridge id<MTLBuffer>)state.range->control_indirect.buffer.get();
    id<MTLBuffer> status =
        (__bridge id<MTLBuffer>)state.range->control_status.buffer.get();
    if (control == nil || count == nil || params == nil || indirect == nil ||
        status == nil) {
      return rund::AccelCheck{false, "accel_metal_command_unavailable"};
    }
    [state.encoder setComputePipelineState:control];
    [state.encoder setBuffer:count
                      offset:static_cast<NSUInteger>(
                                 state.range->control_count.ref.offset_bytes +
                                 state.range->control.count_byte_offset)
                     atIndex:0u];
    [state.encoder setBuffer:params offset:0u atIndex:1u];
    [state.encoder setBuffer:indirect offset:0u atIndex:2u];
    [state.encoder setBuffer:status offset:0u atIndex:3u];
    [state.encoder
              dispatchThreads:MTLSizeMake(state.range->stage_count, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(state.range->stage_count, 1u, 1u)];
    [state.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  }
  for (std::uint32_t index = 0u; index < state.range->stage_count; ++index) {
    const std::optional<RangeParams> params =
        state.range->controlled
            ? std::nullopt
            : RangeStageParamsFor(state.range->range, index);
    if (!state.range->controlled && !params.has_value()) {
      SetMetalLastError(adapter, "compute_range_aggregate_invalid");
      return rund::AccelCheck{false, "compute_range_aggregate_invalid"};
    }
    EncodeMetalRangeStage(state, index,
                          params.has_value() ? &*params : nullptr);
    if (index + 1u < state.range->stage_count &&
        RangeUsesScratch(state.range->range)) {
      [state.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
  }
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  (void)command_encoder;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
