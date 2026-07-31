#include "local.hpp"
#include <rund/counter.hpp>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include <cstddef>
#include <cstring>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
bool UploadMetalBuffer(MetalAdapter& adapter,
                       const MetalRuntimeBuffer& buffer,
                       const void* const data,
                       const rund::kernel::u64 bytes) {
  if (bytes == 0u) {
    return true;
  }
  if (buffer.buffer == nullptr || data == nullptr || bytes > buffer.bytes) {
    return false;
  }
  id<MTLBuffer> metal_buffer = (__bridge id<MTLBuffer>)buffer.buffer.get();
  void* const contents = [metal_buffer contents];
  if (contents == nullptr) {
    return false;
  }
  std::memcpy(static_cast<std::byte *>(contents) + buffer.offset, data,
              static_cast<std::size_t>(bytes));
  std::lock_guard<std::mutex> lock{adapter.mutex};
  ::rund::detail::counter::Accumulate(adapter.stats.host_to_device_bytes,
                                      bytes);
  return true;
}

void* MetalBufferContents(const MetalRuntimeBuffer& buffer) {
  if (buffer.buffer == nullptr) {
    return nullptr;
  }
  id<MTLBuffer> metal_buffer = (__bridge id<MTLBuffer>)buffer.buffer.get();
  void *const contents = [metal_buffer contents];
  return contents == nullptr
             ? nullptr
             : static_cast<void *>(static_cast<std::byte *>(contents) +
                                   buffer.offset);
}
#else
bool UploadMetalBuffer(MetalAdapter&,
                       const MetalRuntimeBuffer&,
                       const void*,
                       rund::kernel::u64) {
  return false;
}

void* MetalBufferContents(const MetalRuntimeBuffer&) {
  return nullptr;
}
#endif

}  // namespace rund::node::accel::detail
