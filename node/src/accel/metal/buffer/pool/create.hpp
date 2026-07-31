#pragma once

#include "../local.hpp"
#include <rund/counter.hpp>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include <algorithm>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

[[nodiscard]] MetalRuntimeBuffer CreateMetalRuntimeBuffer(
    MetalAdapter& adapter,
    const rund::kernel::u64 bytes,
    const MetalBufferUsage usage) {
  NSUInteger length = 0u;
  if (!ToNSUInteger(bytes, length)) {
    return {};
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  if (device == nil) {
    return {};
  }
  id<MTLBuffer> metal_buffer =
      [device newBufferWithLength:length options:MTLResourceStorageModeShared];
  std::shared_ptr<void> handle = RetainMetalObject((__bridge void*)metal_buffer);
  if (handle == nullptr) {
    return {};
  }

  std::lock_guard<std::mutex> lock{adapter.mutex};
  const std::uint64_t id = adapter.next_runtime_buffer_id++;
  ::rund::detail::counter::Accumulate(adapter.stats.buffer_allocation_count,
                                      1u);
  ::rund::detail::counter::Accumulate(adapter.memory.current, bytes);
  adapter.memory.peak = std::max(adapter.memory.peak, adapter.memory.current);
  ::rund::detail::counter::Accumulate(adapter.memory.cumulative, bytes);
  return MetalRuntimeBuffer{.id = id,
                            .bytes = bytes,
                            .usage = usage,
                            .buffer = handle,
                            .reused = false};
}

#endif

}  // namespace rund::node::accel::detail
