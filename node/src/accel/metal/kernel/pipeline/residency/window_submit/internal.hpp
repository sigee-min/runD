#pragma once

#include "../window_submit.hpp"

#include <cstdint>
#include <span>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK) &&                 \
    defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0

extern const std::uint64_t MetalResidencyWindowTerminalDeadlineNs;
extern const std::uint64_t MetalResidencyWindowFaultDeadlineNs;

[[nodiscard]] rund::AccelCheck
PrepareMetalResidencyWindowCommand(MetalSequence &,
                                   MetalResidencyWindowCommand &,
                                   std::span<const std::uint32_t>) noexcept
    API_AVAILABLE(macos(26.0), ios(26.0));

void CancelMetalResidencyWindowCommand(MetalResidencyWindowCommand &) noexcept
    API_AVAILABLE(macos(26.0), ios(26.0));

#endif

} // namespace rund::node::accel::detail
