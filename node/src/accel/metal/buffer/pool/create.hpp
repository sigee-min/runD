#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] MetalRuntimeBuffer
CreateMetalRuntimeBuffer(MetalAdapter &adapter, rund::kernel::u64 bytes,
                         MetalBufferUsage usage);

} // namespace rund::node::accel::detail
