#pragma once

#include "../adapter.hpp"
#include "submit.hpp"

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
struct CommandRun final {
  id<MTLCommandBuffer> buffer = nil;
  id<MTLComputeCommandEncoder> encoder = nil;
};

enum class ResourceRefs : bool {
  Retained,
  Borrowed,
};

template <ResourceRefs Refs = ResourceRefs::Retained>
[[nodiscard]] inline rund::AccelCheck OpenCommand(MetalAdapter &adapter,
                                                  CommandRun &command) {
  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)adapter.queue.get();
  if constexpr (Refs == ResourceRefs::Borrowed) {
    command.buffer =
        queue == nil ? nil : [queue commandBufferWithUnretainedReferences];
  } else {
    command.buffer = queue == nil ? nil : [queue commandBuffer];
  }
  command.encoder =
      command.buffer == nil ? nil : [command.buffer computeCommandEncoder];
  if (command.buffer != nil && command.encoder != nil) {
    return rund::AccelCheck{true, "ok"};
  }
  SetMetalLastError(adapter, "accel_metal_command_unavailable");
  return rund::AccelCheck{false, "accel_metal_command_unavailable"};
}

inline void CloseCommand(const CommandRun &command) {
  [command.encoder endEncoding];
}

[[nodiscard]] inline rund::AccelCheck
FinishCommand(MetalAdapter &adapter, const CommandRun &command,
              const rund::AccelCheck encoded,
              rund::RuntimeStats *const stats = nullptr) {
  CloseCommand(command);
  return encoded.ok
             ? WaitCommand(adapter, (__bridge void *)command.buffer, stats)
             : encoded;
}
#endif

} // namespace rund::node::accel::detail
