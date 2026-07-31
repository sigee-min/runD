#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline const char* GatherFunctionName(
    const rund::kernel::GatherElement element) noexcept {
  return element == rund::kernel::GatherElement::U64
             ? "rund_compute_gather_u64"
             : "rund_compute_gather_u32";
}

[[nodiscard]] inline std::string GatherPipelineKey(
    const rund::kernel::GatherElement element) {
  return element == rund::kernel::GatherElement::U64 ? "gather.u64"
                                                            : "gather.u32";
}
#endif

}  // namespace rund::node::accel::detail
