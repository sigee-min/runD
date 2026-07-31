#pragma once

#include "build.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
inline void StoreMetalScanPipelineSet(MetalAdapter& adapter,
                                      const std::string& block_key,
                                      const std::string& prefix_key,
                                      const std::string& offset_key,
                                      const std::shared_ptr<void>& block,
                                      const std::shared_ptr<void>& prefix,
                                      const std::shared_ptr<void>& offset,
                                      const std::uint64_t create_ns) {
  StoreMetalNamedPipeline(adapter, block_key, block, create_ns);
  StoreMetalNamedPipeline(adapter, prefix_key, prefix, 0u);
  StoreMetalNamedPipeline(adapter, offset_key, offset, 0u);
}
#endif

}  // namespace rund::node::accel::detail
