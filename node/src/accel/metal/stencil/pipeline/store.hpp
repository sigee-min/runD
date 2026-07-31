#pragma once

#include "compile.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
inline void StoreMetalStencilPipeline(MetalAdapter& adapter,
                                      const std::string& key,
                                      std::shared_ptr<void> pipeline,
                                      std::uint64_t compile_ns) {
  StoreMetalNamedPipeline(adapter, key, std::move(pipeline), compile_ns);
}
#endif

}  // namespace rund::node::accel::detail
