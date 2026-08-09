#pragma once

#include "../local.hpp"

#if defined(__OBJC__) && defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelCheck
EnsureMetalPreparedDispatchTrace(id<MTLDevice> device,
                                 MetalKernelResources &resources,
                                 PreparedMemoryMeter *memory) noexcept;
[[nodiscard]] rund::AccelCheck
EncodeMetalPreparedStageTrace(MetalAdapter &adapter,
                              MetalKernelResources &resources,
                              id<MTLDevice> device, CommandRun &run) noexcept;

} // namespace rund::node::accel::detail
#endif
