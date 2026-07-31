#include "model.hpp"

#include "../../backend.hpp"
#include "../../buffer/local.hpp"
#include "../../cpu/run/state.hpp"
#include "../../graph/state.hpp"
#include "../../type.hpp"
#include <rund/counter.hpp>

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

namespace rund::compute::detail {
namespace {

[[nodiscard]] Status prepare_graph_buffers(JobState &state) {
  if (state.program->graph_bindings.empty()) {
    return Status::success();
  }
  std::vector<std::shared_ptr<BufferState>> *buffers = &state.graph_buffers;
  if (state.workspace != nullptr) {
    if (state.workspace->program != state.program) {
      return Status::fail(Reason::PipelineInvalid);
    }
    buffers = &state.workspace->buffers;
  } else if (state.program->device->backend == Backend::Cpu) {
    return Status::success();
  }
  if (!buffers->empty()) {
    if (buffers->size() != state.program->chunks.size() ||
        (state.workspace != nullptr &&
         state.workspace->offsets.size() != buffers->size())) {
      return Status::fail(Reason::PipelineInvalid);
    }
    for (std::size_t index = 0u; index < buffers->size(); ++index) {
      const Chunk chunk = state.program->chunks[index];
      const std::shared_ptr<BufferState> &buffer = (*buffers)[index];
      const std::size_t offset =
          state.workspace == nullptr ? 0u : state.workspace->offsets[index];
      if (buffer == nullptr || buffer->device != state.program->device ||
          buffer->type != Type::U32 || offset > buffer->count ||
          chunk.count > buffer->count - offset) {
        return Status::fail(Reason::PipelineInvalid);
      }
    }
    return Status::success();
  }
  buffers->reserve(state.program->chunks.size());
  for (const Chunk chunk : state.program->chunks) {
    auto made = make_workspace_buffer(state.program->device, chunk.count);
    if (!made) {
      return Status::fail(made.reason());
    }
    buffers->push_back(std::move(made).value());
  }
  return Status::success();
}

} // namespace

Result<RunState> empty_run(const std::shared_ptr<JobState> &state) {
  if (state == nullptr || state->program == nullptr || state->outputs.empty()) {
    return Result<RunState>::fail(Reason::RunInvalid);
  }
  RunState run{};
  run.program = state->program;
  std::copy(state->outputs.begin(), state->outputs.end(), run.outputs.begin());
  run.stats = Stats{.backend = state->program->device->backend,
                    .graph_read_bytes = state->program->graph_info.read_bytes,
                    .graph_hash = state->program->empty_graph_hash};
  return Result<RunState>::success(std::move(run));
}

Status prepare_job_state(const std::shared_ptr<JobState> &state,
                         const JobBindings mode) {
  const Status graph_buffers = prepare_graph_buffers(*state);
  if (!graph_buffers) {
    return graph_buffers;
  }
  const Status cpu_prepared = prepare_cpu_run(*state);
  if (!cpu_prepared) {
    return cpu_prepared;
  }
  const DeviceOps *const ops = state->program->device->ops;
  if (state->program->device->backend != Backend::Cpu &&
      (ops == nullptr || ops->prepare_job == nullptr)) {
    return Status::fail(Reason::DeviceInvalid);
  }
  if (ops == nullptr) {
    return Status::success();
  }
  std::vector<rund::AccelRunBinding> bindings(
      state->program->graph_bindings.size());
  const Status prepared = ops->prepare_job(state, bindings);
  if (!prepared || mode == JobBindings::ReadOnly) {
    return prepared;
  }
  auto active = std::move(state->prepared);
  state->inputs.swap(state->write_inputs);
  const Status pending = ops->prepare_job(state, bindings);
  state->inputs.swap(state->write_inputs);
  if (!pending) {
    return pending;
  }
  state->write_prepared = std::move(state->prepared);
  state->prepared = std::move(active);
  return Status::success();
}

Result<std::shared_ptr<JobState>>
finish_prepare(std::shared_ptr<JobState> state, const Status status) noexcept {
  if (status) {
    return Result<std::shared_ptr<JobState>>::success(std::move(state));
  }
  const std::uint32_t node =
      state == nullptr ? Location::none : state->prepared.failed_node;
  return Result<std::shared_ptr<JobState>>::fail(status.reason(),
                                                 Location{.node = node});
}

Result<std::shared_ptr<JobState>>
make_job_values(const std::shared_ptr<ProgramState> &program,
                const std::span<const HostView> host_inputs,
                const JobBindings bindings) {
  if (program == nullptr || program->device == nullptr) {
    return Result<std::shared_ptr<JobState>>::fail(Reason::ProgramInvalid);
  }
  const Status inputs_valid = validate_host_inputs(*program, host_inputs);
  if (!inputs_valid) {
    return Result<std::shared_ptr<JobState>>::fail(inputs_valid.reason());
  }

  if (program->output_types.empty() ||
      program->output_types.size() != program->output_sizes.size() ||
      program->output_types.size() > MaxOutputs) {
    return Result<std::shared_ptr<JobState>>::fail(
        Reason::BindingCountMismatch);
  }
  try {
    auto state = std::make_shared<JobState>();
    state->terminal = std::make_unique<JobTerminalState>();
    state->program = program;
    state->inputs.reserve(host_inputs.size());
    if (bindings == JobBindings::Writable) {
      state->write_inputs.reserve(host_inputs.size());
    }
    state->outputs.reserve(program->output_types.size());
    for (std::size_t index = 0u; index < host_inputs.size(); ++index) {
      const Type expected_type = program->input_types[index];
      const std::size_t expected_size = program->input_sizes[index];
      auto source = upload_raw(program->device, host_inputs[index]);
      if (!source) {
        return Result<std::shared_ptr<JobState>>::fail(source.reason());
      }
      state->inputs.push_back(std::move(source).value());
      if (bindings == JobBindings::Writable) {
        auto write_source = make_input_binding_buffer(
            program->device, expected_type, expected_size);
        if (!write_source) {
          return Result<std::shared_ptr<JobState>>::fail(write_source.reason());
        }
        state->write_inputs.push_back(std::move(write_source).value());
      }
      const std::uint64_t bytes =
          static_cast<std::uint64_t>(expected_size * type_bytes(expected_type));
      ::rund::detail::counter::Accumulate(state->transfer_bytes, bytes);
      state->transfer_peak = std::max(state->transfer_peak, bytes);
    }
    for (std::size_t index = 0u; index < program->output_types.size();
         ++index) {
      auto output = make_buffer(program->device, program->output_types[index],
                                program->output_sizes[index]);
      if (!output) {
        return Result<std::shared_ptr<JobState>>::fail(output.reason());
      }
      state->outputs.push_back(std::move(output).value());
    }
    const Status prepared = prepare_job_state(state, bindings);
    return finish_prepare(std::move(state), prepared);
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<JobState>>::fail(Reason::BufferCapacity);
  }
}

} // namespace rund::compute::detail
