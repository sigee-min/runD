#include "../number.hpp"
#include "../resident.hpp"
#include "owner.hpp"

#include <kernel/core/checked.hpp>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include <cstdint>

#include <unistd.h>

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] std::uint64_t host_page_bytes() noexcept {
  static const std::uint64_t page = []() noexcept {
    const long value = ::sysconf(_SC_PAGESIZE);
    return value > 0 ? static_cast<std::uint64_t>(value) : 0u;
  }();
  return page;
}

} // namespace

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
  if (requirement.size < length) {
    return 0u;
  }
  const std::uint64_t page = host_page_bytes();
  if (page == 0u) {
    return 0u;
  }
  if (requirement.size <= page) {
    return requirement.size;
  }
  std::uint64_t committed = 0u;
  return kernel::checked::align_up(static_cast<std::uint64_t>(requirement.size),
                                   page, committed)
             ? committed
             : 0u;
#else
  (void)pick;
  (void)logical_bytes;
  return 0u;
#endif
}

} // namespace rund::node::accel::detail
