#include "state.hpp"

#include "../device/info.hpp"
#include "../memory/profile.hpp"

#include <mutex>
#include <utility>

namespace rund::compute::detail {

bool valid_virtual_pipeline(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept {
  return state != nullptr && state->input != nullptr &&
         state->output != nullptr && state->pipeline != nullptr &&
         state->pipeline->residency != nullptr &&
         state->pipeline->residency_staging != nullptr &&
         state->pipeline->residency->identity() &&
         valid_pipeline(state->pipeline);
}

Stats virtual_pipeline_stats(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept {
  if (state == nullptr) {
    return {};
  }
  std::lock_guard lock{state->gate};
  return state->stats;
}

MemoryStats virtual_pipeline_memory(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept {
  if (state == nullptr) {
    return {};
  }
  std::lock_guard lock{state->gate};
  return pipeline_memory(state->pipeline);
}

PipelinePlan virtual_pipeline_plan(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept {
  if (state == nullptr) {
    return {};
  }
  std::lock_guard lock{state->gate};
  return pipeline_plan(state->pipeline);
}

Result<telemetry::Profile> virtual_pipeline_profile(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept {
  if (!valid_virtual_pipeline(state) || state->pipeline->device == nullptr) {
    return Result<telemetry::Profile>::fail(Reason::ProfileInvalid);
  }
  std::unique_lock lock{state->gate, std::try_to_lock};
  if (!lock.owns_lock() || state->phase == VirtualPipelinePhase::Running) {
    return Result<telemetry::Profile>::fail(Reason::ProfileBusy);
  }
  if (state->samples != VirtualPipelineState::SampleState::Inactive) {
    return Result<telemetry::Profile>::fail(Reason::ProfileBusy);
  }
  std::shared_ptr<const DeviceInfo> device =
      device_info_owner(state->pipeline->device);
  if (device == nullptr) {
    return Result<telemetry::Profile>::fail(Reason::DeviceInfoInvalid);
  }
  return Result<telemetry::Profile>::success(ProfileAccess::make(
      std::move(device), state->stats, pipeline_memory(state->pipeline)));
}

} // namespace rund::compute::detail
