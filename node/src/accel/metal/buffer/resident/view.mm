#include <accel/device.hpp>

#include "../../../backend/result.hpp"
#include "../../resident/access.hpp"
#include "find.hpp"

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include <cstddef>
#include <memory>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
BackendHostView
ReadMetalResidentBuffer(const rund::AccelDevice &pick,
                        const rund::kernel::ResidentBufferRef &ref,
                        const std::shared_ptr<void> &handle) noexcept {
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr || handle == nullptr) {
    return {};
  }
  MetalResidentBufferResult resolved =
      ResolvePrivateMetalResidentBuffer(*adapter, ref, handle);
  if (!resolved.check.ok || resolved.device_buffer == nullptr ||
      resolved.ref.offset_bytes > resolved.ref.bytes) {
    return {};
  }
  id<MTLBuffer> metal_buffer =
      (__bridge id<MTLBuffer>)resolved.device_buffer.get();
  const void *const contents = [metal_buffer contents];
  if (contents == nullptr || resolved.ref.bytes == 0u) {
    return {};
  }
  return BackendHostView{
      .data = static_cast<const std::byte *>(contents) +
              static_cast<std::size_t>(resolved.ref.offset_bytes),
      .bytes = resolved.ref.bytes,
  };
}

BackendHostWriteView
WriteMetalResidentBuffer(const rund::AccelDevice &pick,
                         const rund::kernel::ResidentBufferRef &ref,
                         const std::shared_ptr<void> &handle) noexcept {
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr || handle == nullptr) {
    return {};
  }
  MetalResidentBufferResult resolved =
      ResolvePrivateMetalResidentBuffer(*adapter, ref, handle);
  if (!resolved.check.ok || resolved.device_buffer == nullptr ||
      resolved.ref.offset_bytes > resolved.ref.bytes) {
    return {};
  }
  id<MTLBuffer> metal_buffer =
      (__bridge id<MTLBuffer>)resolved.device_buffer.get();
  void *const contents = [metal_buffer contents];
  if (contents == nullptr || resolved.ref.bytes == 0u) {
    return {};
  }
  return BackendHostWriteView{
      .data = static_cast<std::byte *>(contents) +
              static_cast<std::size_t>(resolved.ref.offset_bytes),
      .bytes = resolved.ref.bytes,
  };
}
#else
BackendHostView
ReadMetalResidentBuffer(const rund::AccelDevice &,
                        const rund::kernel::ResidentBufferRef &,
                        const std::shared_ptr<void> &) noexcept {
  return {};
}

BackendHostWriteView
WriteMetalResidentBuffer(const rund::AccelDevice &,
                         const rund::kernel::ResidentBufferRef &,
                         const std::shared_ptr<void> &) noexcept {
  return {};
}
#endif

} // namespace rund::node::accel::detail
