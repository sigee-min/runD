#pragma once

#include "../local.hpp"

#include <algorithm>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] bool
BindInputs(id<MTLComputeCommandEncoder> encoder, const id<MTLBuffer> input,
           const rund::kernel::ComputePlan &plan,
           const rund::kernel::ComputeDispatchWindow &window,
           const rund::kernel::BindingSet &bindings,
           const std::span<const InputWindowPlan> input_plans) {
  if (input_plans.size() != plan.input_buffer_count) {
    return false;
  }
  rund::kernel::u64 cursor = 0u;
  for (rund::kernel::u64 index = 0u; index < plan.input_buffer_count; ++index) {
    if (input == nil) {
      return false;
    }
    rund::kernel::u64 offset = 0u;
    rund::kernel::u64 range = 0u;
    rund::kernel::u64 next = 0u;
    const rund::kernel::ComputeDispatchWindow input_window =
        InputWindow(input_plans[static_cast<std::size_t>(index)], window);
    if (!StagedInputRange(bindings.input_buffers[index], input_window, cursor,
                          1u, offset, range, next)) {
      return false;
    }
    NSUInteger ns_offset = 0u;
    if (!ToNSUInteger(offset, ns_offset)) {
      return false;
    }
    [encoder setBuffer:input
                offset:ns_offset
               atIndex:static_cast<NSUInteger>(index + 1u)];
    cursor = next;
  }
  return true;
}

[[nodiscard]] bool EncodeWindow(
    id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline,
    const rund::kernel::ComputePlan &plan,
    const rund::kernel::ComputeDispatchWindow &window,
    const rund::kernel::BindingSet &bindings, const MetalRuntimeBuffer &param,
    const std::span<const InputWindowPlan> input_plans,
    const MetalRuntimeBuffer *const input, const MetalRuntimeBuffer &output) {
  if (encoder == nil) {
    return false;
  }
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:(__bridge id<MTLBuffer>)param.buffer.get()
              offset:0u
             atIndex:0u];
  id<MTLBuffer> input_buffer =
      input == nullptr ? nil : (__bridge id<MTLBuffer>)input->buffer.get();
  if (!BindInputs(encoder, input_buffer, plan, window, bindings, input_plans)) {
    return false;
  }
  [encoder setBuffer:(__bridge id<MTLBuffer>)output.buffer.get()
              offset:0u
             atIndex:static_cast<NSUInteger>(plan.input_buffer_count + 1u)];
  const NSUInteger thread_count = static_cast<NSUInteger>(window.tile_count);
  const NSUInteger width = std::max<NSUInteger>(
      1u, std::min<NSUInteger>(thread_count,
                               [pipeline maxTotalThreadsPerThreadgroup]));
  [encoder dispatchThreads:MTLSizeMake(thread_count, 1u, 1u)
      threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
  return true;
}

} // namespace
#endif

} // namespace rund::node::accel::detail
