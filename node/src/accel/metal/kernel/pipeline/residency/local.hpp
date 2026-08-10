#pragma once

#include "../state.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

void PrimeMetalResidencySubmission(MetalSequence &sequence,
                                   bool succeeded) noexcept;

[[nodiscard]] KernelResult
RunMetalResidencySubmission(MetalSequence &sequence,
                            KernelTiming timing) noexcept;

#endif

} // namespace rund::node::accel::detail
