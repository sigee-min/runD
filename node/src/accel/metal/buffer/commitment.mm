#include "../number.hpp"
#include "../resident.hpp"
#include "owner.hpp"

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include <cstdint>

namespace rund::node::accel::detail {

std::uint64_t
MetalBufferStorageBytes(const rund::AccelDevice &pick,
                        const std::uint64_t logical_bytes) noexcept {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  NSUInteger length = 0u;
  if (adapter == nullptr || logical_bytes == 0u ||
      !ToNSUInteger(logical_bytes, length)) {
    return 0u;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter->device.get();
  if (device == nil) {
    return 0u;
  }
  const MTLSizeAndAlign requirement =
      [device heapBufferSizeAndAlignWithLength:length
                                       options:MTLResourceStorageModeShared];
  return requirement.size >= length ? requirement.size : 0u;
#else
  (void)pick;
  (void)logical_bytes;
  return 0u;
#endif
}

} // namespace rund::node::accel::detail
