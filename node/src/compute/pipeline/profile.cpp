#include <rund/compute/pipeline.hpp>

#include "../device/info.hpp"
#include "../memory/profile.hpp"
#include "claim.hpp"
#include "local.hpp"
#include "run/clock.hpp"
#include "run/memory.hpp"
#include "state.hpp"

#include <algorithm>
#include <mutex>
#include <span>
#include <utility>

namespace rund::compute::detail {

Result<PipelineProfileSnapshot>
pipeline_profile(const std::shared_ptr<PipelineState> &state,
                 const std::span<PipelineStepProfile> steps) noexcept {
  if (!valid_pipeline(state)) {
    return Result<PipelineProfileSnapshot>::fail(Reason::ProfileInvalid);
  }
  std::unique_lock lock{state->gate, std::try_to_lock};
  if (!lock.owns_lock() || state->phase == PipelinePhase::Running ||
      state->samples != PipelineState::SampleState::Inactive) {
    return Result<PipelineProfileSnapshot>::fail(Reason::ProfileBusy);
  }
  if (state->profile == nullptr) {
    return Result<PipelineProfileSnapshot>::fail(Reason::ProfileUnavailable);
  }
  if (state->publication != nullptr) {
    std::lock_guard publication_lock{state->publication->gate};
    synchronize_pipeline_observation_epoch(*state, *state->publication);
    state->stats.publication.generation = state->publication->generation;
  }
  const std::uint64_t started = pipeline_clock();
  const std::span<PipelineStepProfile> canonical{state->profile->steps.data(),
                                                 state->steps.size()};
  const PipelineMemoryView view =
      pipeline_memory_view_locked(*state, canonical);
  const std::size_t written = std::min(steps.size(), canonical.size());
  std::copy_n(canonical.begin(), written, steps.begin());
  PipelineProfileSnapshot snapshot{
      .execution = state->stats,
      .memory = view.summary,
      .shared_memory = view.shared,
      .referenced_resource_bytes = view.referenced_resource_bytes,
      .instrumentation_command_count =
          state->profile->instrumentation_command_count,
      .instrumentation_byte_count = state->profile->instrumentation_byte_count,
      .written = written,
      .total = canonical.size(),
  };
  const std::uint64_t finished = pipeline_clock();
  snapshot.observation =
      StepTiming{.duration_ns = finished >= started ? finished - started : 0u,
                 .sample_count = 1u,
                 .clock = StepClock::HostSteady,
                 .relation = StepTimingRelation::Exclusive};
  return Result<PipelineProfileSnapshot>::success(snapshot);
}

Result<telemetry::Profile>
pipeline_profile(const std::shared_ptr<PipelineState> &state) noexcept {
  if (!valid_pipeline(state) || state->device == nullptr) {
    return Result<telemetry::Profile>::fail(Reason::ProfileInvalid);
  }
  std::shared_ptr<const DeviceInfo> device = device_info_owner(state->device);
  if (device == nullptr) {
    return Result<telemetry::Profile>::fail(Reason::DeviceInfoInvalid);
  }
  std::unique_lock lock{state->gate, std::try_to_lock};
  if (!lock.owns_lock() || state->phase == PipelinePhase::Running ||
      state->samples != PipelineState::SampleState::Inactive) {
    return Result<telemetry::Profile>::fail(Reason::ProfileBusy);
  }
  if (state->publication != nullptr) {
    std::lock_guard publication_lock{state->publication->gate};
    synchronize_pipeline_observation_epoch(*state, *state->publication);
    state->stats.publication.generation = state->publication->generation;
  }
  const MemoryStats memory = pipeline_memory_view_locked(*state).summary;
  return Result<telemetry::Profile>::success(
      ProfileAccess::make(std::move(device), state->stats, memory));
}

} // namespace rund::compute::detail
