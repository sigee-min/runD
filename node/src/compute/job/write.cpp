#include "write.hpp"
#include <rund/counter.hpp>
#include "control/model.hpp"

#include <algorithm>
#include <utility>

namespace rund::compute::detail {

void record_write(JobState &state, const WriteStats stats) noexcept {
  state.write = stats;
  ::rund::detail::counter::Accumulate(state.transfer_bytes, stats.bytes);
  state.transfer_peak = std::max(state.transfer_peak, stats.bytes);
}

Status write_job_raw(const std::shared_ptr<JobState> &state,
                     const std::span<const HostView> inputs) {
  if (state == nullptr || state->program == nullptr) {
    return Status::fail(Reason::RunInvalid);
  }
  if (!valid_input_shape(*state->program)) {
    return Status::fail(Reason::ProgramInputShapeInvalid);
  }
  const std::size_t expected = state->program->input_types.size();
  if (inputs.size() != expected || state->inputs.size() != expected ||
      state->write_inputs.size() != expected) {
    return Status::fail(Reason::BindingCountMismatch);
  }
  const Status inputs_valid = validate_host_inputs(*state->program, inputs);
  if (!inputs_valid) {
    return inputs_valid;
  }
  {
    std::lock_guard lock{state->gate};
    if (job_busy(state->phase)) {
      return Status::fail(Reason::JobBusy);
    }
    state->phase = JobPhase::Writing;
    state->write = {};
  }

  WriteStats stats{};
  for (std::size_t index = 0u; index < expected; ++index) {
    const Status written =
        write_buffer(state->write_inputs[index], inputs[index], stats);
    if (!written) {
      std::lock_guard lock{state->gate};
      record_write(*state, stats);
      state->phase = JobPhase::Idle;
      return written;
    }
  }
  {
    std::lock_guard lock{state->gate};
    state->inputs.swap(state->write_inputs);
    std::swap(state->prepared, state->write_prepared);
    if (state->terminal != nullptr) {
      state->terminal->last.reset();
    }
    record_write(*state, stats);
    state->phase = JobPhase::Idle;
  }
  return Status::success();
}

WriteStats job_write_stats(const std::shared_ptr<JobState> &state) noexcept {
  if (state == nullptr) {
    return {};
  }
  std::lock_guard lock{state->gate};
  return state->write;
}

} // namespace rund::compute::detail
