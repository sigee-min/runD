#include "../resource.hpp"

#include <kernel/program/compute/transform/stage.hpp>

#include <algorithm>
#include <cstddef>

namespace rund::node::accel::detail {

rund::AccelCheck EncodeMetalNumeric(MetalAdapter &adapter,
                                    const std::shared_ptr<void> &prepared,
                                    void *const command_encoder) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const state = static_cast<MetalNumericPrepared *>(prepared.get());
  id<MTLComputeCommandEncoder> encoder =
      (__bridge id<MTLComputeCommandEncoder>)command_encoder;
  if (state == nullptr || state->adapter != &adapter || encoder == nil ||
      state->pipeline == nullptr || state->buffer_count == 0u ||
      (state->grouped ? state->groups == 0u : state->threads == 0u)) {
    return rund::AccelCheck{false, "accel_metal_command_unavailable"};
  }
  if (state->status_index < state->buffer_count) {
    ClearStatus(state->buffers[state->status_index], state->status_count);
  }
  id<MTLComputePipelineState> pipeline =
      (__bridge id<MTLComputePipelineState>)state->pipeline.get();
  [encoder setComputePipelineState:pipeline];
  for (std::size_t index = 0u; index < state->buffer_count; ++index) {
    [encoder setBuffer:ToMetalBuffer(state->buffers[index])
                offset:static_cast<NSUInteger>(
                           state->buffers[index].ref.offset_bytes)
               atIndex:index];
  }
  const NSUInteger capacity = [pipeline maxTotalThreadsPerThreadgroup];
  if (state->transform_count != 0u) {
    const NSUInteger width =
        static_cast<NSUInteger>(rund::kernel::transform_stage::Lanes);
    if (width > capacity) {
      return rund::AccelCheck{false, "accel_metal_command_unavailable"};
    }
    [encoder setBuffer:state->twiddle.buffer == nullptr
                           ? ToMetalBuffer(state->buffers[0])
                           : (__bridge id<MTLBuffer>)state->twiddle.buffer.get()
                offset:state->twiddle.buffer == nullptr
                           ? static_cast<NSUInteger>(
                                 state->buffers[0].ref.offset_bytes)
                           : 0u
               atIndex:state->buffer_count];
    const auto dispatch = [&](const rund::kernel::transform_stage::Batch batch,
                              const rund::kernel::u64 threads) {
      NumericParams params = state->params;
      params.inner = batch.span;
      params.cols = batch.stride;
      params.rhs_cols = batch.next_span;
      params.value_count = batch.next_stride;
      params.max_iterations = static_cast<rund::kernel::u32>(batch.bit_count);
      [encoder setBytes:&params
                 length:sizeof(params)
                atIndex:state->buffer_count + 1u];
      [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(threads), 1u,
                                           1u)
          threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
    };
    const auto local =
        rund::kernel::transform_stage::Describe(state->transform_count, 1u);
    dispatch(local, rund::kernel::transform_stage::Threads(
                        state->transform_count, local));
    if (state->transform_count == 1u) {
      return rund::AccelCheck{true, "ok"};
    }
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    for (rund::kernel::u64 span =
             rund::kernel::transform_stage::FirstGlobalSpan;
         span != 0u && span <= state->transform_count;) {
      const auto batch =
          rund::kernel::transform_stage::Describe(state->transform_count, span);
      dispatch(batch, rund::kernel::transform_stage::Threads(
                          state->transform_count, batch));
      span = rund::kernel::transform_stage::Next(state->transform_count, batch);
      if (span == 0u) {
        break;
      }
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    return rund::AccelCheck{true, "ok"};
  }
  [encoder setBytes:&state->params
             length:sizeof(state->params)
            atIndex:state->buffer_count];
  if (state->grouped) {
    const NSUInteger width = static_cast<NSUInteger>(state->threadgroup);
    if (width == 0u || width > capacity) {
      return rund::AccelCheck{false, "accel_metal_command_unavailable"};
    }
    [encoder dispatchThreadgroups:MTLSizeMake(
                                      static_cast<NSUInteger>(state->groups),
                                      1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
  } else {
    const NSUInteger width = std::max<NSUInteger>(
        1u, std::min<NSUInteger>(static_cast<NSUInteger>(state->threadgroup),
                                 capacity));
    [encoder dispatchThreads:MTLSizeMake(
                                 static_cast<NSUInteger>(state->threads), 1u,
                                 1u)
        threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
  }
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)prepared;
  (void)command_encoder;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
