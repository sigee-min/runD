#pragma once

#include <rund/compute/status.hpp>

namespace rund::compute::detail {

struct VirtualPipelineState;

// One mutation-free capability query for the two prepared residency banks.
// Unsupported means not ready; other errors retain their terminal identity.
[[nodiscard]] Status
ready_virtual_residency_window(const VirtualPipelineState &) noexcept;

} // namespace rund::compute::detail
