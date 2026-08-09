#pragma once

#include "../../../pipeline/named.hpp"
#include "../../source.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline bool BuildMetalScanPipelineSet(
    id<MTLDevice> device, id<MTLLibrary> library, NSString *block_name,
    NSString *prefix_name, NSString *offset_name, const bool with_offsets,
    std::shared_ptr<void> &block, std::shared_ptr<void> &prefix,
    std::shared_ptr<void> &offset) {
  return (block != nullptr ||
          MakeNamedMetalPipeline(device, library, block_name, block)) &&
         (!with_offsets ||
          ((prefix != nullptr ||
            MakeNamedMetalPipeline(device, library, prefix_name, prefix)) &&
           (offset != nullptr ||
            MakeNamedMetalPipeline(device, library, offset_name, offset))));
}
#endif

} // namespace rund::node::accel::detail
