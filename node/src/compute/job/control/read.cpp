#include "model.hpp"

#include <rund/counter.hpp>
#include "../../program/output.hpp"
#include "../../type.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace rund::compute::detail {
Result<std::size_t> job_read_size_impl(const std::shared_ptr<JobState> &state,
                                       const std::size_t logical_output,
                                       const Type expected_type) {
  if (state == nullptr || state->program == nullptr) {
    return Result<std::size_t>::fail(Reason::RunInvalid);
  }
  std::lock_guard lock{state->gate};
  if (job_busy(state->phase)) {
    return Result<std::size_t>::fail(Reason::JobRunning);
  }
  if (job_failed(state->phase)) {
    return Result<std::size_t>::fail(state->failure);
  }
  if (state->terminal == nullptr || !state->terminal->last.has_value()) {
    return Result<std::size_t>::fail(Reason::ResidentNotRun);
  }
  const std::size_t physical_output =
      output_index(state->program->output_aliases, logical_output);
  if (physical_output >= state->outputs.size() ||
      state->outputs[physical_output] == nullptr ||
      state->outputs[physical_output]->type != expected_type) {
    return Result<std::size_t>::fail(Reason::ShapeMismatch);
  }
  return Result<std::size_t>::success(state->outputs[physical_output]->count);
}

Status job_read_data_impl(const std::shared_ptr<JobState> &state,
                          const std::size_t logical_output,
                          const Type expected_type, void *const data,
                          const std::size_t bytes,
                          const std::size_t logical_count,
                          const bool destination_zeroed) {
  if (state == nullptr || state->program == nullptr) {
    return Status::fail(Reason::RunInvalid);
  }
  std::lock_guard lock{state->gate};
  if (job_busy(state->phase)) {
    return Status::fail(Reason::JobRunning);
  }
  if (job_failed(state->phase)) {
    return Status::fail(state->failure);
  }
  if (state->terminal == nullptr || !state->terminal->last.has_value()) {
    return Status::fail(Reason::ResidentNotRun);
  }
  const std::size_t physical_output =
      output_index(state->program->output_aliases, logical_output);
  if (physical_output >= state->outputs.size() ||
      state->outputs[physical_output] == nullptr ||
      state->outputs[physical_output]->type != expected_type) {
    return Status::fail(Reason::ShapeMismatch);
  }
  const auto &buffer = state->outputs[physical_output];
  const std::size_t read_count =
      logical_count == std::numeric_limits<std::size_t>::max() ? buffer->count
                                                               : logical_count;
  if (read_count > buffer->count ||
      bytes != read_count * type_bytes(expected_type) ||
      (data == nullptr && bytes != 0u)) {
    return Status::fail(Reason::ShapeMismatch);
  }
  std::uint64_t staging_bytes = 0u;
  std::uint64_t staging_budget = 0u;
  bool staging_reused = false;
  const Status status = read_job_buffer(
      *state->terminal->last, buffer, data, bytes, read_count, staging_bytes,
      staging_reused, staging_budget, destination_zeroed);
  state->staging_peak = std::max(state->staging_peak, staging_bytes);
  state->staging_bytes = ::rund::detail::counter::SaturatingAdd(
      state->staging_bytes, staging_bytes);
  state->staging_reused = ::rund::detail::counter::SaturatingAdd(
      state->staging_reused, staging_reused ? staging_bytes : 0u);
  state->staging_budget = std::max(state->staging_budget, staging_budget);
  if (!status) {
    return status;
  }
  if (!state->program->empty()) {
    ::rund::detail::counter::Accumulate(state->transfer_bytes, bytes);
    state->transfer_peak =
        std::max(state->transfer_peak, static_cast<std::uint64_t>(bytes));
  }
  return Status::success();
}

Result<std::size_t> job_read_size(const std::shared_ptr<JobState> &state,
                                  const std::size_t output, const Type type) {
  return job_read_size_impl(state, output, type);
}

Status job_read_data(const std::shared_ptr<JobState> &state,
                     const std::size_t output, const Type type,
                     void *const data, const std::size_t bytes,
                     const std::size_t logical_count,
                     const bool destination_zeroed) {
  return job_read_data_impl(state, output, type, data, bytes, logical_count,
                            destination_zeroed);
}

} // namespace rund::compute::detail
