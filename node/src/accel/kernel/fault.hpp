#pragma once

#include <accel/device.hpp>

namespace rund::node::accel::detail {

// Contract-only native completion fault. The next Metal/Vulkan command is
// still submitted and retired normally; only its terminal result is projected
// as DeviceLost. This exercises pending-state discard without fabricating a
// pre-submit rejection or switching backend.
[[nodiscard]] bool
InjectNativeDeviceLostOnce(const rund::AccelDevice &pick) noexcept;

} // namespace rund::node::accel::detail
