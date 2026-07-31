#pragma once

#include "create.hpp"
#include "select.hpp"
#include "../../scratch.hpp"
#include <algorithm>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

MetalRuntimeBuffer AcquireMetalBuffer(MetalAdapter& adapter,
                                      const rund::kernel::u64 bytes,
                                      const MetalBufferUsage usage) {
  const rund::kernel::u64 stored_bytes = std::max<rund::kernel::u64>(1u, bytes);
  if (usage == MetalBufferUsage::Scratch) {
    MetalScratch *const scratch = ActiveMetalScratch();
    if (scratch != nullptr) {
      return scratch->acquire(stored_bytes);
    }
  }
  const MetalBufferUsage physical =
      usage == MetalBufferUsage::Scratch ? MetalBufferUsage::Output : usage;
  MetalRuntimeBuffer reused =
      TakeReusableMetalBuffer(adapter, stored_bytes, physical);
  return reused.buffer != nullptr
             ? reused
             : CreateMetalRuntimeBuffer(adapter, stored_bytes, physical);
}

#else

MetalRuntimeBuffer AcquireMetalBuffer(MetalAdapter&,
                                      rund::kernel::u64,
                                      MetalBufferUsage) {
  return {};
}

#endif

}  // namespace rund::node::accel::detail
