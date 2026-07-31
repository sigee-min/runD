#pragma once

#include "model.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
[[nodiscard]] inline bool StoreResident(VulkanResidentBufferResult* const dst,
                                        const VulkanResidentBufferResult& src) {
  if (dst == nullptr) { return false; }
  dst->check = src.check;
  dst->ref = src.ref;
  dst->handle = src.handle;
  dst->storage = src.storage;
  dst->device_buffer = src.device_buffer;
  dst->storage_bytes = src.storage_bytes;
  dst->storage_reused = src.storage_reused;
  return true;
}
#endif

}  // namespace rund::node::accel::detail
