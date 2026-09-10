#pragma once

#include "../state.hpp"
#include "../../../kernel.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

// Raw bounded-window lifecycle owner.  The public declarations live here so
// its prepare/submit/signal/abort protocol remains separate from the single
// warm submit and from the canonical command encoder.
[[nodiscard]] rund::AccelCheck
SubmitMetalResidencyWindow(const BackendResidencyWindowRequest &) noexcept;
[[nodiscard]] rund::AccelCheck
SignalMetalResidencyWindow(const std::shared_ptr<void> &,
                           const BackendResidencyWindowSignal &) noexcept;
[[nodiscard]] rund::AccelCheck
AbortMetalResidencyWindow(const std::shared_ptr<void> &,
                          const BackendResidencyWindowAbort &) noexcept;

#endif

} // namespace rund::node::accel::detail
