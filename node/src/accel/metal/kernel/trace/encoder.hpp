#pragma once

#include "../trace.hpp"

#if defined(__OBJC__) && defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace rund::node::accel::detail {

[[nodiscard]] id CreateMetalDispatchTraceEncoder() noexcept;
void BindMetalDispatchTraceEncoder(id proxy,
                                   id<MTLComputeCommandEncoder> encoder,
                                   MetalDispatchTrace &trace) noexcept;
void ClearMetalDispatchTraceEncoder(id proxy) noexcept;

} // namespace rund::node::accel::detail
#endif
