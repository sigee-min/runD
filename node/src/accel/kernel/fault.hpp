#pragma once

#include "fault/domain.hpp"

#include <accel/device.hpp>

namespace rund::node::accel::detail {

// Contract-only native completion fault. The next Work command is still
// submitted and retired normally; only its terminal result is projected as
// DeviceLost. This exercises pending-state discard without fabricating a
// pre-submit rejection or switching backend.
[[nodiscard]] bool
InjectNativeDeviceLostOnce(const rund::AccelDevice &pick) noexcept;

// Contract-only native transfer fault. The next synchronous transfer command
// is still submitted and retired normally; only its terminal result is
// projected as DeviceLost.
[[nodiscard]] bool InjectNativeTransferDeviceLostOnce(
    const rund::AccelDevice &pick) noexcept;

// Contract-only private-transfer fault. The next nonempty accelerator
// download fails before copying bytes. This proves Device-to-Host migration
// rollback without fabricating an Authority transition or backing failure.
[[nodiscard]] bool
InjectNativeDownloadFailureOnce(const rund::AccelDevice &pick) noexcept;

// Contract-only coherent-view capability fault. The next exact HostRead view
// request is denied without failing the physical download route. This proves
// that Direct execution can fall back to the ordinary Device-to-Host
// migration and then return to the coherent route on the same prepared owner.
[[nodiscard]] bool
InjectNativeHostReadUnavailableOnce(const rund::AccelDevice &pick) noexcept;

// Contract-only coherent-supply capability fault. The next exact HostWrite
// view request is denied without touching the ordinary upload route. This
// proves capability fallback and same-owner return independently of transfer
// failure behavior.
[[nodiscard]] bool
InjectNativeHostWriteUnavailableOnce(const rund::AccelDevice &pick) noexcept;

// Contract-only queue-terminal loss. The next Metal residency command is
// submitted without its queue event or callback. Its bounded deadline must
// wake failure, quarantine every may-write owner, and make the Device's VSM
// capability sticky-unavailable without fabricating success.
[[nodiscard]] bool
InjectNativeResidencyTerminalLossOnce(const rund::AccelDevice &pick) noexcept;

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
