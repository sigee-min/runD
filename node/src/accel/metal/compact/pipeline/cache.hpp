#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
inline void LookupMetalCompactPipelines(MetalAdapter& adapter,
                                        MetalCompactPipelines& pipelines,
                                        const bool status_required,
                                        const bool block_offsets) {
  if (block_offsets) {
    pipelines.count_blocks =
        LookupMetalNamedPipeline(adapter, "compact.count_blocks");
    pipelines.scatter_blocks =
        LookupMetalNamedPipeline(adapter, "compact.scatter_blocks");
  } else {
    pipelines.scatter = LookupMetalNamedPipeline(adapter, "compact.scatter");
  }
  if (status_required) {
    pipelines.status = LookupMetalNamedPipeline(adapter, "compact.status");
  }
}

[[nodiscard]] inline bool MetalCompactPipelinesReady(
    const MetalCompactPipelines& pipelines,
    const bool status_required,
    const bool block_offsets) {
  return block_offsets
             ? pipelines.count_blocks != nullptr &&
                   pipelines.scatter_blocks != nullptr
             : pipelines.scatter != nullptr &&
                   (!status_required || pipelines.status != nullptr);
}
#endif

}  // namespace rund::node::accel::detail
