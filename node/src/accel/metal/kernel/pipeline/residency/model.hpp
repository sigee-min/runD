#pragma once

#include <accel/check.hpp>

#include <cstdint>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

enum class MetalResidencySubmissionPhase : std::uint8_t {
  Unsupported,
  Cold,
  Ready,
  Failed,
};

struct MetalResidencySubmission final {
#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  id<MTL4CommandQueue> queue = nil;
  id<MTL4CommandAllocator> allocator = nil;
  id<MTL4CommandBuffer> command = nil;
  id<MTLResidencySet> resources = nil;
  id<MTLSharedEvent> event = nil;
#endif
  MetalResidencySubmissionPhase phase =
      MetalResidencySubmissionPhase::Unsupported;
  std::uint64_t event_value{};

  [[nodiscard]] bool supported() const noexcept {
    return phase != MetalResidencySubmissionPhase::Unsupported;
  }

  [[nodiscard]] bool ready() const noexcept {
    return phase == MetalResidencySubmissionPhase::Ready;
  }
};

#endif

} // namespace rund::node::accel::detail
