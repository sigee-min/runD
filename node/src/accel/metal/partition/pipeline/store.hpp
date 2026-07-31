#pragma once

#include "compile.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
inline void StoreMetalPartitionPipeline(MetalAdapter& adapter,
                                        const char* const key,
                                        const std::shared_ptr<void>& pipeline,
                                        const std::uint64_t elapsed_ns) {
  StoreMetalNamedPipeline(adapter, key, pipeline, elapsed_ns);
}
#endif

}  // namespace rund::node::accel::detail
