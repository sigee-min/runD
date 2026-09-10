#pragma once

#include "local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0

// One canonical selected-local lowering surface.  Warm submission, bounded
// Window submission, Sliding, and Schedule all consume this encoder so range
// accounting and barrier placement cannot drift between routes.
[[nodiscard]] rund::AccelCheck EncodeMetalResidencySelection(
    id<MTL4ComputeCommandEncoder>, MetalSequence &,
    std::span<const std::uint32_t>, std::uint64_t &dispatch_count,
    std::uint64_t &control_count, std::uint64_t &reset_count,
    std::uint64_t &reset_bytes) noexcept API_AVAILABLE(macos(26.0), ios(26.0));

[[nodiscard]] id<MTL4CommandQueue>
MetalResidencyQueue(const MetalSequence &) noexcept
    API_AVAILABLE(macos(26.0), ios(26.0));

#endif

#endif

} // namespace rund::node::accel::detail
