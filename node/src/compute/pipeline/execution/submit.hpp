#pragma once

#include "model.hpp"
#include "prepare.hpp"

#include "../../device/residency/execution/plan.hpp"

#include <memory>
#include <span>

namespace rund::node::accel::detail {
struct PreparedPipelineEvidence;
}

namespace rund::compute::detail {

// Shared terminal validation/publication for both selected epoch submission
// and the Q=1 run-level native owner.
[[nodiscard]] PipelineOutcome finish_residency_pipeline_execution(
    PipelineState &, std::span<const std::uint32_t>,
    const node::accel::detail::PreparedPipelineEvidence &,
    PipelineOutcome) noexcept;
[[nodiscard]] Status
publish_residency_pipeline_execution(PipelineState &, std::size_t,
                                     const PipelineOutcome &) noexcept;

// Begins the existing PrivateResidency Pipeline attempt, submits the exact
// Plan-projected locals through the raw selected backend seam, validates its
// PreparedPipelineEvidence, publishes the Pipeline terminal, and only then
// emits NativeEvidence. Synchronous submit rejection returns failure and emits
// no native callback because no backend command was accepted.
[[nodiscard]] Status submit_pipeline_execution(
    const std::shared_ptr<PipelineState> &, const residency::execution::Owner &,
    const residency::execution::Plan &, residency::execution::Control &,
    PipelineExecutionSubmission &) noexcept;

} // namespace rund::compute::detail
