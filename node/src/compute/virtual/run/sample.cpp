#include "../local.hpp"

#include <mutex>

namespace rund::compute::detail {

Status begin_virtual_pipeline_samples(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept {
  if (!valid_virtual_pipeline(state)) {
    return Status::fail(Reason::ProfileInvalid);
  }
  std::unique_lock lock{state->gate, std::try_to_lock};
  if (!lock.owns_lock() || state->phase == VirtualPipelinePhase::Running) {
    return Status::fail(Reason::ProfileBusy);
  }
  if (state->phase == VirtualPipelinePhase::Poisoned) {
    return Status::fail(Reason::PipelinePoisoned);
  }
  if (state->samples != VirtualPipelineState::SampleState::Inactive) {
    return Status::fail(Reason::ProfileBusy);
  }
  state->stats.pipeline.residency.sampled_runs = 0u;
  state->stats.pipeline.residency.allocation_free_runs = 0u;
  state->samples = VirtualPipelineState::SampleState::Active;
  return Status::success();
}

Status end_virtual_pipeline_samples(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept {
  if (!valid_virtual_pipeline(state)) {
    return Status::fail(Reason::ProfileInvalid);
  }
  std::unique_lock lock{state->gate, std::try_to_lock};
  if (!lock.owns_lock() || state->phase == VirtualPipelinePhase::Running) {
    return Status::fail(Reason::ProfileBusy);
  }
  if (state->phase == VirtualPipelinePhase::Poisoned) {
    return Status::fail(Reason::PipelinePoisoned);
  }
  if (state->samples == VirtualPipelineState::SampleState::Inactive) {
    return Status::fail(Reason::ProfileInvalid);
  }
  state->samples = VirtualPipelineState::SampleState::Inactive;
  return Status::success();
}

} // namespace rund::compute::detail
