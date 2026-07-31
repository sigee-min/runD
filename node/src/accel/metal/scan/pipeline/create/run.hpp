#pragma once

#include "lookup.hpp"
#include "store.hpp"
#include "../../../../clock.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline bool CompileMetalScanPipelineSet(
    MetalAdapter& adapter,
    NSString* block_name,
    NSString* prefix_name,
    NSString* offset_name,
    const std::string& block_key,
    const std::string& prefix_key,
    const std::string& offset_key,
    std::shared_ptr<void>& block,
    std::shared_ptr<void>& prefix,
    std::shared_ptr<void>& offset) {
  if (LookupMetalScanPipelineSet(adapter, block_key, prefix_key, offset_key,
                                 block, prefix, offset)) {
    return true;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  const std::uint64_t create_begin = MonotonicNanoseconds();
  std::shared_ptr<void> library_owner =
      AcquireMetalLibrary(adapter, MetalScanSource());
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)library_owner.get();
  if (!BuildMetalScanPipelineSet(device, library, block_name, prefix_name,
                                 offset_name, block, prefix, offset)) {
    return false;
  }
  StoreMetalScanPipelineSet(adapter, block_key, prefix_key, offset_key, block,
                            prefix, offset,
                            MonotonicNanoseconds() - create_begin);
  return true;
}
#endif

}  // namespace rund::node::accel::detail
