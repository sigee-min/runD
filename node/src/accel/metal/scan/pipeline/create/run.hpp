#pragma once

#include "../../../../clock.hpp"
#include "lookup.hpp"
#include "store.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline bool CompileMetalScanPipelineSet(
    MetalAdapter &adapter, NSString *block_name, NSString *prefix_name,
    NSString *offset_name, const std::string &block_key,
    const std::string &prefix_key, const std::string &offset_key,
    const bool with_offsets, std::shared_ptr<void> &block,
    std::shared_ptr<void> &prefix, std::shared_ptr<void> &offset) {
  if (LookupMetalScanPipelineSet(adapter, block_key, prefix_key, offset_key,
                                 with_offsets, block, prefix, offset)) {
    return true;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  const std::uint64_t create_begin = MonotonicNanoseconds();
  std::shared_ptr<void> library_owner =
      AcquireMetalLibrary(adapter, MetalScanSource());
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)library_owner.get();
  const bool build_block = block == nullptr;
  const bool build_prefix = with_offsets && prefix == nullptr;
  const bool build_offset = with_offsets && offset == nullptr;
  if (!BuildMetalScanPipelineSet(device, library, block_name, prefix_name,
                                 offset_name, with_offsets, block, prefix,
                                 offset)) {
    return false;
  }
  return StoreMetalScanPipelineSet(adapter, block_key, prefix_key, offset_key,
                                   build_block, build_prefix, build_offset,
                                   block, prefix, offset,
                                   MonotonicNanoseconds() - create_begin);
}
#endif

} // namespace rund::node::accel::detail
