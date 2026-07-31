#pragma once

#include "name.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline bool LookupMetalScatterPipeline(
    MetalAdapter& adapter,
    const std::string& key,
    std::shared_ptr<void>& out) {
  out = LookupMetalNamedPipeline(adapter, key);
  return out != nullptr;
}
#endif

}  // namespace rund::node::accel::detail
