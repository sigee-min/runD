#pragma once

#include "../../adapter.hpp"
#include "../../state.hpp"
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
static constexpr std::size_t kInlineMetalBufferCount = 4u;

class ScopedMetalBuffers {
 public:
  explicit ScopedMetalBuffers(MetalAdapter& adapter) : adapter_(adapter) {}
  ScopedMetalBuffers(const ScopedMetalBuffers&) = delete;
  ScopedMetalBuffers& operator=(const ScopedMetalBuffers&) = delete;
  ~ScopedMetalBuffers() {
    for (std::size_t index = 0u; index < inline_count_; ++index) {
      ReleaseMetalBuffer(adapter_, std::move(inline_buffers_[index]));
    }
    for (MetalRuntimeBuffer& buffer : overflow_buffers_) {
      ReleaseMetalBuffer(adapter_, std::move(buffer));
    }
  }

  [[nodiscard]] MetalRuntimeBuffer& add(MetalRuntimeBuffer buffer) {
    if (inline_count_ < kInlineMetalBufferCount) {
      inline_buffers_[inline_count_] = std::move(buffer);
      return inline_buffers_[inline_count_++];
    }
    overflow_buffers_.push_back(std::move(buffer));
    return overflow_buffers_.back();
  }

 private:
  MetalAdapter& adapter_;
  std::array<MetalRuntimeBuffer, kInlineMetalBufferCount> inline_buffers_{};
  std::vector<MetalRuntimeBuffer> overflow_buffers_{};
  std::size_t inline_count_ = 0u;
};
#endif

}  // namespace rund::node::accel::detail
