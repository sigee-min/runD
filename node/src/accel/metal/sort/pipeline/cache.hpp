#pragma once

#include "name.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
inline void LookupMetalSortPipelines(MetalAdapter &adapter,
                                     const std::string &dispatch_key,
                                     const std::string &histogram_key,
                                     const std::string &prefix_key,
                                     const std::string &base_key,
                                     const std::string &scatter_key,
                                     MetalSortPipelines &pipelines) {
  pipelines.dispatch = LookupMetalNamedPipeline(adapter, dispatch_key);
  pipelines.histogram = LookupMetalNamedPipeline(adapter, histogram_key);
  pipelines.prefix = LookupMetalNamedPipeline(adapter, prefix_key);
  pipelines.base = LookupMetalNamedPipeline(adapter, base_key);
  pipelines.scatter = LookupMetalNamedPipeline(adapter, scatter_key);
}
#endif

} // namespace rund::node::accel::detail
