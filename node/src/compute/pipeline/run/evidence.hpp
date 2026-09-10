#pragma once

#include "result.hpp"

namespace rund::compute::detail {

// Immutable input: no Pipeline, device, publication, lock or lease is mutated
// by evidence decisions. active_steps means native extent for ordinary runs
// and the exact issued-local extent for Residency.
struct PipelineEvidenceContext final {
  std::size_t total_steps{};
  std::size_t active_steps{};
  std::uint64_t generation{};
  bool prior_submission{};
  bool timed_metal{};
};

struct PipelineEvidenceDecision final {
  PipelineOutcome outcome{};
  bool layout_valid{};
  bool control_valid{};
  PipelineNestedPhase failed_nested_phase{};
};

[[nodiscard]] PipelineEvidenceDecision decide_ordinary_pipeline_evidence(
    const PipelineEvidenceContext &context,
    const node::accel::detail::PreparedPipelineEvidence &evidence) noexcept;
[[nodiscard]] PipelineEvidenceDecision decide_residency_pipeline_evidence(
    const PipelineEvidenceContext &context,
    const node::accel::detail::PreparedPipelineEvidence &evidence,
    PipelineOutcome prepared = {}) noexcept;

} // namespace rund::compute::detail
