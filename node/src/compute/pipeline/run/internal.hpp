#pragma once

#include "../local.hpp"
#include "result.hpp"

#include "../../../accel/kernel/prepared/interface/api.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace rund::compute::detail {

struct DeviceOps;

// Backend/path seams shared by the public pipeline orchestrator and the
// execution owners.  The implementations live in exactly one path owner;
// this header carries only the private cross-TU contract.
[[nodiscard]] PipelineOutcome run_cpu(PipelineState &, std::size_t step_limit);
[[nodiscard]] PipelineOutcome
run_accel(PipelineState &, node::accel::detail::PipelineSubmitMode mode);
[[nodiscard]] PipelineOutcome
run_cpu_residency_steps(PipelineState &, std::span<const std::uint32_t> locals);

[[nodiscard]] Status residency_pipeline_locals(
    const std::shared_ptr<PipelineState> &, residency::EpochLease,
    std::span<std::uint32_t> local_order, std::size_t &issued_steps) noexcept;

struct ResidencyBackendSelection final {
  const DeviceOps *ops{};
  const node::accel::detail::PreparedKernelPipeline *prepared{};
};

[[nodiscard]] Status publish_residency_pipeline_outcome(
    PipelineState &, std::size_t issued_steps, const PipelineOutcome &,
    bool defer_generation, std::uint64_t publication_generation,
    std::uint8_t publication_parity) noexcept;

[[nodiscard]] Status prepare_residency_pipeline_submission(
    const std::shared_ptr<PipelineState> &, residency::EpochLease,
    ResidencyPipelineSubmission &, ResidencyPipelineCompletion, void *user,
    ResidencyBackendSelection &, bool defer_generation,
    std::uint64_t control_generation, std::uint8_t control_parity,
    std::uint64_t publication_generation,
    std::uint8_t publication_parity) noexcept;

void fail_residency_pipeline_submission(ResidencyPipelineSubmission &,
                                        Status failure) noexcept;
void complete_residency_pipeline_submission(
    void *, node::accel::detail::PreparedPipelineEvidence &&) noexcept;

} // namespace rund::compute::detail
