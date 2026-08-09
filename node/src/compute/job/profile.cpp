#include "state.hpp"

#include "../backend.hpp"
#include "../device/info.hpp"
#include "../memory/local.hpp"
#include "../memory/profile.hpp"

#include <rund/compute/abi/observe.hpp>

#include <mutex>
#include <utility>

namespace rund::compute::detail {

Result<telemetry::Profile>
job_profile(const std::shared_ptr<JobState> &state) noexcept {
  if (state == nullptr || state->program == nullptr ||
      state->program->device == nullptr) {
    return Result<telemetry::Profile>::fail(Reason::ProfileInvalid);
  }
  std::shared_ptr<const DeviceInfo> device =
      device_info_owner(state->program->device);
  if (device == nullptr) {
    return Result<telemetry::Profile>::fail(Reason::DeviceInfoInvalid);
  }
  std::lock_guard lock{state->gate};
  if (job_busy(state->phase)) {
    return Result<telemetry::Profile>::fail(Reason::ProfileBusy);
  }
  const Stats execution = job_stats_locked(*state);
  return Result<telemetry::Profile>::success(ProfileAccess::make(
      std::move(device), execution, job_memory_locked(*state)));
}

} // namespace rund::compute::detail
