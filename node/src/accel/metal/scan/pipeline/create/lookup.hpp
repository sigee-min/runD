#pragma once

#include "../../../pipeline/cache.hpp"
#include "../name.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline bool LookupMetalScanPipelineSet(
    MetalAdapter &adapter, const std::string &block_key,
    const std::string &prefix_key, const std::string &offset_key,
    const bool with_offsets, std::shared_ptr<void> &block,
    std::shared_ptr<void> &prefix, std::shared_ptr<void> &offset) {
  block = LookupMetalNamedPipeline(adapter, block_key);
  prefix = with_offsets ? LookupMetalNamedPipeline(adapter, prefix_key)
                        : std::shared_ptr<void>{};
  offset = with_offsets ? LookupMetalNamedPipeline(adapter, offset_key)
                        : std::shared_ptr<void>{};
  return block != nullptr &&
         (!with_offsets || (prefix != nullptr && offset != nullptr));
}
#endif

} // namespace rund::node::accel::detail
