#include "model.hpp"

#include "../../pipeline/claim.hpp"
#include "../../program/output.hpp"
#include "../../size.hpp"
#include "../../type.hpp"
#include "../local.hpp"
#include "../write.hpp"

#include <array>
#include <memory>
#include <utility>

namespace rund::compute::detail {
namespace {

[[nodiscard]] std::shared_ptr<ProgramState>
borrow_program(ProgramState *const program) noexcept {
  return std::shared_ptr<ProgramState>{std::shared_ptr<ProgramState>{},
                                       program};
}

} // namespace

Status refresh_host(const std::shared_ptr<JobState> &state,
                    const std::span<const HostView> inputs) noexcept {
  if (state == nullptr || state->program == nullptr) {
    return Status::fail(Reason::RunInvalid);
  }
  const Status valid = validate_host_inputs(*state->program, inputs);
  if (!valid) {
    return valid;
  }
  if (state->inputs.size() != inputs.size()) {
    return Status::fail(Reason::BindingCountMismatch);
  }

  {
    std::lock_guard lock{state->gate};
    if (job_busy(state->phase)) {
      return Status::fail(Reason::JobBusy);
    }
    state->phase = JobPhase::Writing;
  }
  WriteStats stats{};
  for (std::size_t index = 0u; index < inputs.size(); ++index) {
    const Status written =
        write_buffer(state->inputs[index], inputs[index], stats);
    if (!written) {
      std::lock_guard lock{state->gate};
      record_write(*state, stats);
      state->phase = JobPhase::Idle;
      return written;
    }
  }
  std::lock_guard lock{state->gate};
  state->terminal->last.reset();
  state->terminal->failed_stats.reset();
  record_write(*state, stats);
  state->phase = JobPhase::Idle;
  return Status::success();
}

Result<std::shared_ptr<JobState>>
run_transient(const std::shared_ptr<ProgramState> &program,
              const std::span<const HostView> inputs) {
  auto state = make_job_values(program, inputs, JobBindings::ReadOnly);
  if (!state) {
    return state;
  }
  const Status status = run_job(*state);
  if (!status) {
    return Result<std::shared_ptr<JobState>>::fail(status.reason());
  }
  return state;
}

Status run_into(const std::shared_ptr<ProgramState> &program,
                const std::span<const HostView> inputs, const Type output_type,
                void *const output_data, const std::size_t output_bytes,
                const std::size_t output_count) {
  if (program == nullptr || program->device == nullptr) {
    return Status::fail(Reason::ProgramInvalid);
  }
  const std::size_t physical_output = output_index(program->output_aliases, 0u);
  const std::size_t element_bytes = type_bytes(output_type);
  std::size_t expected_bytes = 0u;
  if (element_bytes == 0u || physical_output >= program->output_types.size() ||
      physical_output >= program->output_sizes.size() ||
      program->output_types[physical_output] != output_type ||
      program->output_sizes[physical_output] != output_count ||
      !size::multiply(output_count, element_bytes, expected_bytes) ||
      output_bytes != expected_bytes ||
      (output_data == nullptr && output_bytes != 0u)) {
    return Status::fail(Reason::ShapeMismatch);
  }

  std::lock_guard lock{program->cache.gate};
  if (program->cache.job == nullptr ||
      program->cache.input != CacheInput::Host) {
    auto made = make_job_values(program, inputs, JobBindings::ReadOnly);
    if (!made) {
      return Status::fail(made.reason());
    }
    program->cache.job = std::move(made).value();
    program->cache.job->program = borrow_program(program.get());
    program->cache.input = CacheInput::Host;
  } else {
    const Status refreshed = refresh_host(program->cache.job, inputs);
    if (!refreshed) {
      return refreshed;
    }
  }
  const Status ran = run_job(program->cache.job);
  if (!ran) {
    return ran;
  }
  return job_read_data_impl(program->cache.job, 0u, output_type, output_data,
                            output_bytes, output_count, true);
}

Result<RunState>
run_buffers(const std::shared_ptr<ProgramState> &program,
            const std::span<const std::shared_ptr<BufferState>> inputs,
            const std::span<const std::shared_ptr<BufferState>> outputs) {
  const Status valid = validate_bound_buffers(program, inputs, outputs, false);
  if (!valid) {
    return Result<RunState>::fail(valid.reason());
  }
  std::array<BufferClaim, MaxMapInputs + MaxOutputs> claim_storage{};
  std::size_t claim_count = 0u;
  for (const auto &input : inputs) {
    claim_storage[claim_count++] = BufferClaim{input.get(), false};
  }
  for (const auto &output : outputs) {
    claim_storage[claim_count++] = BufferClaim{output.get(), true};
  }
  const std::span<const BufferClaim> claims{claim_storage.data(), claim_count};
  const Status claimed = acquire_claims(*program->device, claims);
  if (!claimed) {
    return Result<RunState>::fail(claimed.reason());
  }
  ClaimGuard claim_guard{*program->device, claims};
  std::lock_guard cache_lock{program->cache.gate};
  if (program->cache.job == nullptr ||
      program->cache.input != CacheInput::Buffers ||
      !same_buffers(program->cache.job, inputs, outputs)) {
    auto made = bind_job_validated(program, inputs, outputs);
    if (!made) {
      return Result<RunState>::fail(made.reason(), made.location());
    }
    program->cache.job = std::move(made).value();
    program->cache.job->program = borrow_program(program.get());
    program->cache.input = CacheInput::Buffers;
  }
  const Status status = run_job(program->cache.job);
  if (!status) {
    publish_claims(*program->device, claims, false, true);
    claim_guard.dismiss();
    return Result<RunState>::fail(status.reason());
  }
  RunState result{};
  bool result_valid = false;
  {
    std::lock_guard job_lock{program->cache.job->gate};
    if (program->cache.job->terminal != nullptr &&
        program->cache.job->terminal->last.has_value()) {
      result = *program->cache.job->terminal->last;
      result_valid = true;
    }
  }
  if (!result_valid) {
    publish_claims(*program->device, claims, false, true);
    claim_guard.dismiss();
    return Result<RunState>::fail(Reason::RunInvalid);
  }
  publish_claims(*program->device, claims, true, false);
  claim_guard.dismiss();
  result.program = program;
  return Result<RunState>::success(std::move(result));
}

} // namespace rund::compute::detail
