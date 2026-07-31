#pragma once

#include "compile.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
inline void StoreMetalCompactPipelines(MetalAdapter& adapter,
                                       const MetalCompactPipelines& pipelines,
                                       const bool status_required,
                                       const bool block_offsets,
                                       const std::uint64_t create_ns) {
  if (block_offsets) {
    StoreMetalNamedPipeline(adapter, "compact.count_blocks",
                            pipelines.count_blocks, create_ns);
    StoreMetalNamedPipeline(adapter, "compact.scatter_blocks",
                            pipelines.scatter_blocks, 0u);
  } else {
    StoreMetalNamedPipeline(adapter, "compact.scatter", pipelines.scatter,
                            create_ns);
  }
  if (status_required) {
    StoreMetalNamedPipeline(adapter, "compact.status", pipelines.status, 0u);
  }
}
#endif

}  // namespace rund::node::accel::detail
