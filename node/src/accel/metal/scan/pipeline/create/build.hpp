#pragma once

#include "../../source.hpp"
#include "../../../pipeline/named.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline bool BuildMetalScanPipelineSet(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString* block_name,
    NSString* prefix_name,
    NSString* offset_name,
    std::shared_ptr<void>& block,
    std::shared_ptr<void>& prefix,
    std::shared_ptr<void>& offset) {
  return MakeNamedMetalPipeline(device, library, block_name, block) &&
         MakeNamedMetalPipeline(device, library, prefix_name, prefix) &&
         MakeNamedMetalPipeline(device, library, offset_name, offset);
}
#endif

}  // namespace rund::node::accel::detail
