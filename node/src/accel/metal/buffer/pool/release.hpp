#pragma once

#include "../local.hpp"
#include <rund/counter.hpp>
#include <utility>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {
constexpr std::size_t kMaxReusableMetalBuffers = 32u;
}

void ReleaseMetalBuffer(MetalAdapter& adapter, MetalRuntimeBuffer buffer) {
  if (buffer.borrowed) {
    return;
  }
  if (buffer.buffer == nullptr || buffer.bytes == 0u) {
    return;
  }
  std::lock_guard<std::mutex> lock{adapter.mutex};
  ::rund::detail::counter::Release(adapter.memory.current, buffer.bytes);
  MetalBuffer released{.id = buffer.id,
                       .bytes = buffer.bytes,
                       .usage = buffer.usage,
                       .buffer = std::move(buffer.buffer)};
  if (adapter.free_buffers.size() < kMaxReusableMetalBuffers) {
    adapter.free_buffers.push_back(std::move(released));
    return;
  }

  auto smallest = adapter.free_buffers.end();
  for (auto it = adapter.free_buffers.begin(); it != adapter.free_buffers.end();
       ++it) {
    if (it->usage != buffer.usage || it->bytes >= buffer.bytes) {
      continue;
    }
    if (smallest == adapter.free_buffers.end() ||
        it->bytes < smallest->bytes) {
      smallest = it;
    }
  }
  if (smallest != adapter.free_buffers.end()) {
    *smallest = MetalBuffer{.id = released.id,
                            .bytes = released.bytes,
                            .usage = released.usage,
                            .buffer = std::move(released.buffer)};
  }
}

#else

void ReleaseMetalBuffer(MetalAdapter&, MetalRuntimeBuffer) {}

#endif

}  // namespace rund::node::accel::detail
