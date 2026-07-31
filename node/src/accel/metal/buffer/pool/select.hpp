#pragma once

#include "../local.hpp"
#include <rund/counter.hpp>

#include <algorithm>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

[[nodiscard]] MetalRuntimeBuffer TakeReusableMetalBuffer(
    MetalAdapter& adapter,
    const rund::kernel::u64 bytes,
    const MetalBufferUsage usage) {
  std::lock_guard<std::mutex> lock{adapter.mutex};
  auto best = adapter.free_buffers.end();
  for (auto it = adapter.free_buffers.begin();
       it != adapter.free_buffers.end(); ++it) {
    if (it->usage != usage || it->bytes < bytes || it->buffer == nullptr) {
      continue;
    }
    if (best == adapter.free_buffers.end() || it->bytes < best->bytes) {
      best = it;
    }
  }
  if (best == adapter.free_buffers.end()) {
    return {};
  }
  MetalRuntimeBuffer buffer{.id = best->id,
                            .bytes = best->bytes,
                            .usage = best->usage,
                            .buffer = best->buffer,
                            .reused = true};
  adapter.free_buffers.erase(best);
  ::rund::detail::counter::Accumulate(adapter.stats.buffer_reuse_hit_count, 1u);
  ::rund::detail::counter::Accumulate(adapter.memory.current, buffer.bytes);
  adapter.memory.peak = std::max(adapter.memory.peak, adapter.memory.current);
  ::rund::detail::counter::Accumulate(adapter.memory.cumulative, buffer.bytes);
  ::rund::detail::counter::Accumulate(adapter.memory.reused, buffer.bytes);
  return buffer;
}

#endif

}  // namespace rund::node::accel::detail
