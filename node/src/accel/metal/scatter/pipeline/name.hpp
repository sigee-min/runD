#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline const char* ScatterFunctionName(
    const rund::kernel::ScatterElement element) noexcept {
  return element == rund::kernel::ScatterElement::U64
             ? "rund_compute_scatter_u64"
             : "rund_compute_scatter_u32";
}

[[nodiscard]] inline std::string ScatterPipelineKey(
    const rund::kernel::ScatterElement element) {
  return element == rund::kernel::ScatterElement::U64 ? "scatter.u64"
                                                             : "scatter.u32";
}
#endif

}  // namespace rund::node::accel::detail
