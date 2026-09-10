#pragma once

#include "../state.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

[[nodiscard]] bool
ObserveMetalControl(MetalSequence &, PreparedPipelineBackendEvidence &) noexcept;
[[nodiscard]] bool
ObserveMetalProfile(MetalSequence &, PreparedPipelineBackendEvidence &) noexcept;
void CompleteMetalSequenceTrace(void *, KernelResult) noexcept;

#endif

} // namespace rund::node::accel::detail
