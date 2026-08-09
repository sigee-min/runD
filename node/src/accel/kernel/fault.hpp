#pragma once

#include <accel/device.hpp>

namespace rund::node::accel::detail {

// Contract-only native completion fault. The next Metal/Vulkan command is
// still submitted and retired normally; only its terminal result is projected
// as DeviceLost. This exercises pending-state discard without fabricating a
// pre-submit rejection or switching backend.
[[nodiscard]] bool
InjectNativeDeviceLostOnce(const rund::AccelDevice &pick) noexcept;

// Contract-only capability fault. The next cold accelerator Trace admission
// rejects before queue admission and before publishing any lazy query/counter
// resources. A retry observes the real immutable device capability.
[[nodiscard]] bool
InjectNativeTraceUnavailableOnce(const rund::AccelDevice &pick) noexcept;

// Contract-only Metal Trace terminal fault. The sampled command completes;
// the following resolve-only command is then reported as DeviceLost.
[[nodiscard]] bool
InjectNativeTraceResolveDeviceLostOnce(const rund::AccelDevice &pick) noexcept;

} // namespace rund::node::accel::detail
