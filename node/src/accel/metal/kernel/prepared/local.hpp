#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] bool PrepareMetalResets(const rund::AccelDevice &pick,
                                      MetalAdapter &adapter,
                                      const BoundResets *resets,
                                      MetalKernelResources &resources);

void CompleteMetalPrepared(void *raw, KernelResult submitted) noexcept;
void CompleteMetalPreparedTrace(void *raw, KernelResult submitted) noexcept;
#endif

} // namespace rund::node::accel::detail
