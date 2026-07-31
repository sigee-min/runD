#include "model.hpp"

#include "../local.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace rund::compute::detail {

Result<std::shared_ptr<JobState>> bind_job_validated(
    const std::shared_ptr<ProgramState> &program,
    const std::span<const std::shared_ptr<BufferState>> inputs,
    const std::span<const std::shared_ptr<BufferState>> outputs) {
  try {
    auto state = std::make_shared<JobState>();
    state->terminal = std::make_unique<JobTerminalState>();
    state->program = program;
    state->inputs.assign(inputs.begin(), inputs.end());
    state->outputs.assign(outputs.begin(), outputs.end());
    const Status prepared = prepare_job_state(state, JobBindings::ReadOnly);
    return finish_prepare(std::move(state), prepared);
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<JobState>>::fail(Reason::BufferCapacity);
  }
}

Result<std::shared_ptr<JobState>>
bind_job(const std::shared_ptr<ProgramState> &program,
         const std::span<const std::shared_ptr<BufferState>> inputs,
         const std::span<const std::shared_ptr<BufferState>> outputs) {
  const Status valid = validate_bound_buffers(program, inputs, outputs);
  return valid ? bind_job_validated(program, inputs, outputs)
               : Result<std::shared_ptr<JobState>>::fail(valid.reason());
}

namespace {

Result<std::shared_ptr<JobState>>
prepare_owned_job(const std::shared_ptr<ProgramState> &program,
                  std::vector<std::shared_ptr<BufferState>> inputs,
                  std::vector<std::shared_ptr<BufferState>> outputs,
                  std::vector<JobBufferView> input_views,
                  std::vector<JobBufferView> output_views,
                  node::accel::detail::KernelViewLayout views,
                  std::shared_ptr<JobWorkspace> workspace) {
  if (program == nullptr || program->device == nullptr) {
    return Result<std::shared_ptr<JobState>>::fail(Reason::ProgramInvalid);
  }
  if ((!input_views.empty() && input_views.size() != inputs.size()) ||
      (!output_views.empty() && output_views.size() != outputs.size())) {
    return Result<std::shared_ptr<JobState>>::fail(
        Reason::BindingCountMismatch);
  }
  try {
    auto state = std::make_shared<JobState>();
    state->program = program;
    state->workspace = std::move(workspace);
    state->inputs = std::move(inputs);
    state->outputs = std::move(outputs);
    state->input_views = std::move(input_views);
    state->output_views = std::move(output_views);
    state->views = std::move(views);
    const Status views = prepare_cpu_view_transfers(*state);
    if (!views) {
      return Result<std::shared_ptr<JobState>>::fail(views.reason());
    }
    const Status prepared = prepare_job_state(state, JobBindings::ReadOnly);
    return finish_prepare(std::move(state), prepared);
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<JobState>>::fail(Reason::BufferCapacity);
  }
}

} // namespace

Result<std::shared_ptr<JobState>>
make_job_raw(const std::shared_ptr<ProgramState> &program,
             const std::span<const HostView> inputs) {
  return make_job_values(program, inputs, JobBindings::Writable);
}

Result<std::shared_ptr<JobState>>
bind_job_buffers(const std::shared_ptr<ProgramState> &program,
                 const std::span<const std::shared_ptr<BufferState>> inputs,
                 const std::span<const std::shared_ptr<BufferState>> outputs) {
  return bind_job(program, inputs, outputs);
}

Result<std::shared_ptr<JobState>>
prepare_pipeline_job_buffers(const std::shared_ptr<ProgramState> &program,
                             std::vector<std::shared_ptr<BufferState>> inputs,
                             std::vector<std::shared_ptr<BufferState>> outputs,
                             std::vector<JobBufferView> input_views,
                             std::vector<JobBufferView> output_views,
                             node::accel::detail::KernelViewLayout views,
                             std::shared_ptr<JobWorkspace> workspace) {
  return prepare_owned_job(program, std::move(inputs), std::move(outputs),
                           std::move(input_views), std::move(output_views),
                           std::move(views), std::move(workspace));
}

} // namespace rund::compute::detail
