#include "../../../accel/kernel/prepared/run.hpp"
#include "prepare.hpp"
#include "../../backend.hpp"

#include "../state.hpp"

#include <rund/counter.hpp>

namespace rund::compute::detail {

PipelineOutcome prepare_residency_pipeline_execution(
    PipelineState &state,
    const std::span<const std::uint32_t> locals) noexcept {
  PipelineOutcome outcome{.publication_suppressed = true};
  const DeviceOps *const ops =
      state.device == nullptr ? nullptr : state.device->ops;
  if (ops == nullptr || locals.empty() || locals.size() > state.steps.size()) {
    outcome.status = Status::fail(Reason::PipelineInvalid);
    return outcome;
  }
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
        job == nullptr || !job->prepared.ok || step.program == nullptr) {
      outcome.status = Status::fail(Reason::PipelineInvalid);
      outcome.failed_step = local;

      return outcome;
    }
    ::rund::detail::counter::Accumulate(state.stats.graph_read_bytes,
                                        step.program->graph_info.read_bytes);
    outcome.writes_possible = outcome.writes_possible || step.writes;
  }
  const node::accel::detail::PreparedKernelPipeline &prepared =
      state.transactional && state.attempt.parity != 0u
          ? state.alternate_prepared
          : state.prepared;
  if (!prepared.ok) {
    outcome.status = Status::fail(Reason::PipelineInvalid);
  }
  return outcome;
}

} // namespace rund::compute::detail
