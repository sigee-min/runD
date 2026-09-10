#include "local.hpp"

#include "../../device/info.hpp"
#include "../../memory/profile.hpp"
#include "../../pipeline/local.hpp"

#include <mutex>
#include <utility>

namespace rund::compute::detail {

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
  return Result<telemetry::Profile>::success(
      ProfileAccess::make(std::move(device), state->stats,
                          virtual_observe::virtual_memory_locked(*state)));
}

} // namespace rund::compute::detail
