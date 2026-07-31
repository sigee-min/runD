#include "model.hpp"

#include <rund/counter.hpp>
#include "../../cpu/run/state.hpp"

#include <rund/compute/abi/observe.hpp>

#include <algorithm>

namespace rund::compute::detail {

Result<Backend> job_backend(const std::shared_ptr<JobState> &state) noexcept {
  return state == nullptr || state->program == nullptr ||
                 state->program->device == nullptr
             ? Result<Backend>::fail(Reason::RunInvalid)
             : Result<Backend>::success(state->program->device->backend);
}

kernel::u32 job_workers(const std::shared_ptr<JobState> &state) noexcept {
  if (state == nullptr || state->program == nullptr ||
      state->program->device == nullptr) {
    return 0u;
  }
  const CpuDeviceState *const cpu = cpu_device(*state->program->device);
  return cpu == nullptr ? 0u : cpu->workers.requested_worker_width;
}

Stats job_stats(const std::shared_ptr<JobState> &state) noexcept {
  if (state == nullptr || state->program == nullptr ||
      state->program->device == nullptr) {
    return {};
  }
  std::lock_guard lock{state->gate};
  if (state->terminal != nullptr && state->terminal->last.has_value()) {
    return run_stats(*state->terminal->last);
  }
  if (state->terminal != nullptr && state->terminal->failed_stats.has_value()) {
    return *state->terminal->failed_stats;
  }
  return Stats{.backend = state->program->device->backend};
}

void record_job_frame(const std::shared_ptr<JobState> &state,
                      const std::uint64_t bytes, const bool reused,
                      const std::uint64_t budget) noexcept {
  if (state == nullptr || bytes == 0u) {
    return;
  }
  std::lock_guard lock{state->gate};
  ::rund::detail::counter::Accumulate(state->frame_current, bytes);
  state->frame_peak = std::max(state->frame_peak, state->frame_current);
  ::rund::detail::counter::Accumulate(state->frame_bytes, bytes);
  ::rund::detail::counter::Accumulate(state->frame_reused, reused ? bytes : 0u);
  state->frame_budget = budget;
}

void release_job_frame(const std::shared_ptr<JobState> &state,
                       const std::uint64_t bytes) noexcept {
  if (state == nullptr || bytes == 0u) {
    return;
  }
  std::lock_guard lock{state->gate};
  ::rund::detail::counter::Release(state->frame_current, bytes);
}

} // namespace rund::compute::detail
