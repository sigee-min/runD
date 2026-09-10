#include "../internal.hpp"

#include "../../../backend.hpp"
#include "../../../job/local.hpp"
#include "../../execution/prepare.hpp"
#include "../../execution/submit.hpp"
#include "../../state.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>

namespace rund::compute::detail {

PipelineOutcome
run_cpu_residency_steps(PipelineState &state,
                        const std::span<const std::uint32_t> locals) {
  PipelineOutcome outcome{.publication_suppressed = true};
  for (const std::uint32_t local : locals) {
    if (local >= state.steps.size()) {
      outcome.status = Status::fail(Reason::PipelineInvalid);
      return outcome;
    }
    PipelineStep &step = state.steps[local];
    const std::shared_ptr<JobState> &job =
        state.transactional && state.attempt.parity != 0u ? step.alternate_job
                                                          : step.job;
    if (step.route != PipelineRoute::Ordinary || step.window != 0u ||
        job == nullptr) {
      outcome.status = Status::fail(Reason::PipelineInvalid);
      outcome.failed_step = local;

      return outcome;
    }
    bool active = true;
    const Status ready = prepare_cpu_pipeline_window(state, local, active);
    if (!ready || !active) {
      outcome.status = ready ? Status::fail(Reason::PipelineInvalid) : ready;
      outcome.failed_step = local;

      return outcome;
    }
    begin_pipeline_profile_step(state, local);
    const Status gathered = gather_cpu_pipeline_views(job);
    Status executed = gathered;
    if (executed) {
      executed = run_pipeline_job(job);
    }
    if (executed) {
      executed = publish_cpu_pipeline_views(job);
    }
    const Status consumed = consume_cpu_pipeline_step(state, local, executed);
    capture_cpu_pipeline_step(state, local, static_cast<bool>(gathered));
    finish_pipeline_profile_step(state, local);
    outcome.writes_possible = outcome.writes_possible || step.writes;
    if (!executed || !consumed) {
      outcome.status = executed ? consumed : executed;
      outcome.failed_step = local;

      return outcome;
    }
    ++outcome.verified;
  }
  state.stats.command_submits = 0u;
  state.stats.pipeline.verified_step_count = outcome.verified;
  state.stats.pipeline.failed_step_index = PipelineStats::no_failed_step;
  return outcome;
}

Status run_residency_pipeline_lease(
    const std::shared_ptr<PipelineState> &state,
    const residency::EpochLease lease, const bool defer_generation,
    const std::uint64_t control_generation, const std::uint8_t control_parity,
    const std::uint64_t publication_generation,
    const std::uint8_t publication_parity) noexcept {
  std::array<std::uint32_t, PipelineLeafCapacity> local_order{};
  std::size_t issued_steps = 0u;
  const Status projected = residency_pipeline_locals(
      state, lease, std::span<std::uint32_t>{local_order}, issued_steps);
  if (!projected) {
    return projected;
  }
  if (state->device->backend != Backend::Cpu) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::unique_lock pipeline_lock{state->gate, std::try_to_lock};
  if (!pipeline_lock.owns_lock()) {
    return Status::fail(Reason::PipelineBusy);
  }
  constexpr PipelineClaimAuthority claim_authority =
      PipelineClaimAuthority::PrivateResidency;
  const Status started = start_pipeline(*state, claim_authority,
                                        control_generation, control_parity);
  if (!started) {
    return started;
  }
  const std::span<const std::uint32_t> issued_locals{local_order.data(),
                                                     issued_steps};
  const PipelineOutcome executed =
      run_cpu_residency_steps(*state, issued_locals);
  return defer_generation ? publish_residency_pipeline_outcome(
                                *state, issued_steps, executed, true,
                                publication_generation, publication_parity)
                          : publish_residency_pipeline_execution(
                                *state, issued_steps, executed);
}

} // namespace rund::compute::detail
