#pragma once

#include "../trace.hpp"

namespace rund::node::accel::detail {

#if defined(__OBJC__) && defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] rund::AccelCheck
QueueMetalDispatchTrace(MetalAdapter &adapter,
                        id<MTLCommandBuffer> sampled_command,
                        MetalDispatchTrace &trace, KernelCompletion completion,
                        void *user) noexcept;
#endif

} // namespace rund::node::accel::detail
