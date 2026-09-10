#pragma once

#include "../run/result.hpp"

#include <cstdint>
#include <span>

namespace rund::compute::detail {

struct PipelineState;

// Shared by the selected epoch executor and the Q=1 run-level owner. This is
// the sole Pipeline-side validation of exact private residency locals.
[[nodiscard]] PipelineOutcome
prepare_residency_pipeline_execution(PipelineState &,
                                     std::span<const std::uint32_t>) noexcept;

} // namespace rund::compute::detail
