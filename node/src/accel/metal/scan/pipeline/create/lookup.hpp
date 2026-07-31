#pragma once

#include "../name.hpp"
#include "../../../pipeline/cache.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline bool LookupMetalScanPipelineSet(
    MetalAdapter& adapter,
    const std::string& block_key,
    const std::string& prefix_key,
    const std::string& offset_key,
    std::shared_ptr<void>& block,
    std::shared_ptr<void>& prefix,
    std::shared_ptr<void>& offset) {
  block = LookupMetalNamedPipeline(adapter, block_key);
  prefix = LookupMetalNamedPipeline(adapter, prefix_key);
  offset = LookupMetalNamedPipeline(adapter, offset_key);
  return block != nullptr && prefix != nullptr && offset != nullptr;
}
#endif

}  // namespace rund::node::accel::detail
