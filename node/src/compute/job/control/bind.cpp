#include "model.hpp"

#include "../../exception.hpp"
#include "../local.hpp"

#include <memory>
#include <new>
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
    const Status prepared = prepare_job_state(state, JobBindings::ReadOnly,
                                              JobGraphBufferMode::Standalone);
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
finish_cpu_pipeline_job(std::shared_ptr<JobState> state,
                        std::shared_ptr<CpuGraphStorage> cpu_storage,
                        const CpuRunRoutePlan *const cpu_route,
                        std::shared_ptr<CpuPreparedArena> prepared_arena,
                        const CpuRunRouteSlice *const cpu_route_slice,
                        const CpuViewTransferLayout &cpu_views) {
  const Status views = prepare_cpu_view_transfers(*state, &cpu_views);
  if (!views) {
    return Result<std::shared_ptr<JobState>>::fail(views.reason());
  }
  Status prepared = Status::success();
  if (cpu_route == nullptr) {
    if (cpu_storage != nullptr || cpu_route_slice != nullptr) {
      prepared = Status::fail(Reason::PipelineInvalid);
    } else {
      prepared = prepare_job_state(state, JobBindings::ReadOnly,
                                   JobGraphBufferMode::SealedPipeline);
    }
  } else if (cpu_route_slice == nullptr) {
    prepared = Status::fail(Reason::PipelineInvalid);
  } else {
    prepared = prepare_cpu_pipeline_job_state(
        state, JobBindings::ReadOnly, std::move(cpu_storage), *cpu_route,
        std::move(prepared_arena), *cpu_route_slice);
  }
  return finish_prepare(std::move(state), prepared);
}

Result<std::shared_ptr<JobState>>
prepare_accel_job(const std::shared_ptr<ProgramState> &program,
                  std::vector<std::shared_ptr<BufferState>> inputs,
                  std::vector<std::shared_ptr<BufferState>> outputs,
                  std::vector<JobBufferView> input_views,
                  std::vector<JobBufferView> output_views,
                  node::accel::detail::KernelViewLayout views,
                  std::shared_ptr<JobWorkspace> workspace) {
  if (program == nullptr || program->device == nullptr ||
      program->device->backend == Backend::Cpu) {
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
    const Status prepared = prepare_job_state(
        state, JobBindings::ReadOnly, JobGraphBufferMode::SealedPipeline);
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
prepare_pipeline_accel_job(const std::shared_ptr<ProgramState> &program,
                           std::vector<std::shared_ptr<BufferState>> inputs,
                           std::vector<std::shared_ptr<BufferState>> outputs,
                           std::vector<JobBufferView> input_views,
                           std::vector<JobBufferView> output_views,
                           node::accel::detail::KernelViewLayout views,
                           std::shared_ptr<JobWorkspace> workspace) {
  return prepare_accel_job(program, std::move(inputs), std::move(outputs),
                           std::move(input_views), std::move(output_views),
                           std::move(views), std::move(workspace));
}

Result<std::shared_ptr<JobState>>
prepare_pipeline_cpu_job(const std::shared_ptr<ProgramState> &program,
                         const CpuJobBindingStorage bindings,
                         std::shared_ptr<CpuPreparedArena> prepared_arena,
                         std::shared_ptr<JobWorkspace> workspace,
                         std::shared_ptr<CpuGraphStorage> cpu_storage,
                         const CpuRunRoutePlan *const cpu_route,
                         const CpuRunRouteSlice *const cpu_route_slice,
                         const CpuViewTransferLayout &cpu_views) {
  if (program == nullptr || program->device == nullptr ||
      program->device->backend != Backend::Cpu || prepared_arena == nullptr ||
      bindings.inputs.size() != cpu_views.input_count ||
      bindings.outputs.size() != cpu_views.output_count ||
      bindings.input_views.size() != bindings.inputs.size() ||
      bindings.output_views.size() != bindings.outputs.size() ||
      bindings.input_transfers.size() != cpu_views.inputs.size() ||
      bindings.output_transfers.size() != cpu_views.outputs.size()) {
    return Result<std::shared_ptr<JobState>>::fail(Reason::PipelineInvalid);
  }
  try {
    auto state = std::make_shared<JobState>();
    state->program = program;
    state->workspace = std::move(workspace);
    state->cpu_prepared_arena = prepared_arena;
    const bool bound =
        state->inputs.bind(bindings.inputs, bindings.inputs.size()) &&
        state->outputs.bind(bindings.outputs, bindings.outputs.size()) &&
        state->input_views.bind(bindings.input_views,
                                bindings.input_views.size()) &&
        state->output_views.bind(bindings.output_views,
                                 bindings.output_views.size()) &&
        state->views.bind(bindings.kernel_views,
                          bindings.kernel_views.size()) &&
        state->cpu_view_inputs.bind(bindings.input_transfers) &&
        state->cpu_view_outputs.bind(bindings.output_transfers);
    if (!bound) {
      return Result<std::shared_ptr<JobState>>::fail(Reason::PipelineInvalid);
    }
    return finish_cpu_pipeline_job(std::move(state), std::move(cpu_storage),
                                   cpu_route, std::move(prepared_arena),
                                   cpu_route_slice, cpu_views);
  } catch (...) {
    compute_exception::rethrow_unless_capacity_exception();
    return Result<std::shared_ptr<JobState>>::fail(Reason::BufferCapacity);
  }
}

} // namespace rund::compute::detail
