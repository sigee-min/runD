#pragma once

#include "compile.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
inline void StoreMetalSortPipelines(
    MetalAdapter &adapter, const std::string &dispatch_key,
    const std::string &histogram_key, const std::string &prefix_key,
    const std::string &base_key, const std::string &scatter_key,
    const MetalSortPipelines &pipelines, const std::uint64_t create_ns) {
  StoreMetalNamedPipeline(adapter, dispatch_key, pipelines.dispatch, create_ns);
  StoreMetalNamedPipeline(adapter, histogram_key, pipelines.histogram, 0u);
  StoreMetalNamedPipeline(adapter, prefix_key, pipelines.prefix, 0u);
  StoreMetalNamedPipeline(adapter, base_key, pipelines.base, 0u);
  StoreMetalNamedPipeline(adapter, scatter_key, pipelines.scatter, 0u);
}
#endif

} // namespace rund::node::accel::detail
