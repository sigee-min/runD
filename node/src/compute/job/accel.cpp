#include "../backend.hpp"
#include "../stats.hpp"
#include "../status.hpp"
#include "state.hpp"

#include <accel/check.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/run/binding.hpp>
#include <kernel/core/checked.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string_view>

namespace rund::compute::detail {
namespace {

[[nodiscard]] Status prepare_arena(const std::shared_ptr<JobState> &state) {
  if (state == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (state->views.empty() &&
      (state->workspace == nullptr || state->workspace->arena == nullptr ||
       state->workspace->arena->scratch.empty())) {
    return Status::success();
  }
  if (state->workspace == nullptr || state->workspace->arena == nullptr ||
      state->program == nullptr || state->program->device == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  JobArena &arena = *state->workspace->arena;
  std::lock_guard lock{arena.gate};
  if (!arena.bound) {
    node::accel::detail::RunBinds owners;
    owners.reserve(arena.buffers.size());
    for (const std::shared_ptr<BufferState> &buffer : arena.buffers) {
      if (buffer == nullptr || buffer->device != state->program->device) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const AccelBufferState *const resident = accel_buffer(*buffer);
      std::shared_ptr<void> handle{};
      const DeviceOps *const ops = state->program->device->ops;
      if (resident == nullptr || ops == nullptr ||
          ops->resolve_buffer == nullptr ||
          !ops->resolve_buffer(*state->program->device, *buffer, handle) ||
          handle == nullptr ||
          !owners.push(resident->buffer.resident, std::move(handle))) {
        return Status::fail(Reason::BindingDeviceMismatch);
      }
    }
    if (!owners.valid() || owners.size() != arena.buffers.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    arena.binds.reserve(arena.slots.size());
    for (const JobArenaSlot slot : arena.slots) {
      if (slot.words == 0u || slot.owner >= owners.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      std::uint64_t offset = 0u;
      std::uint64_t bytes = 0u;
      if (!kernel::checked::mul(
              static_cast<std::uint64_t>(slot.offset_words),
              static_cast<std::uint64_t>(sizeof(std::uint32_t)), offset) ||
          !kernel::checked::mul(
              static_cast<std::uint64_t>(slot.words),
              static_cast<std::uint64_t>(sizeof(std::uint32_t)), bytes)) {
        return Status::fail(Reason::PipelineCapacity);
      }
      rund::kernel::ResidentBufferRef ref = owners.refs()[slot.owner];
      if (offset > ref.bytes || bytes > ref.bytes - offset ||
          ref.offset_bytes >
              std::numeric_limits<std::uint64_t>::max() - offset) {
        return Status::fail(Reason::PipelineInvalid);
      }
      ref.offset_bytes += offset;
      ref.element_bytes = sizeof(std::uint32_t);
      ref.stride_bytes = sizeof(std::uint32_t);
      ref.count = slot.words;
      if (!arena.binds.push(ref, owners.handles()[slot.owner])) {
        return Status::fail(Reason::PipelineCapacity);
      }
    }
    if (!arena.binds.valid() || arena.binds.size() != arena.slots.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    arena.bound = true;
  }
  for (const node::accel::detail::KernelViewSlot &view : state->views) {
    if (view.slot >= arena.binds.size() ||
        arena.binds.refs()[view.slot].offset_bytes >
            arena.binds.refs()[view.slot].bytes ||
        view.bytes > arena.binds.refs()[view.slot].bytes -
                         arena.binds.refs()[view.slot].offset_bytes) {
      return Status::fail(Reason::PipelineInvalid);
    }
  }
  if (!node::accel::detail::ValidKernelScratch(arena.scratch, arena.binds)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  return Status::success();
}

[[nodiscard]] JobBufferView binding_view(const JobState &state,
                                         const GraphRunBinding &binding,
                                         const BufferState &buffer) noexcept {
  return job_binding_view(state, binding, buffer);
}

struct RunBindings final {
  std::span<const rund::AccelRunBinding> graph{};
  Status status = Status::fail(Reason::GraphBindingInvalid);

  [[nodiscard]] explicit operator bool() const noexcept { return status.ok(); }

  [[nodiscard]] const rund::AccelRunBinding *data() const noexcept {
    return graph.data();
  }
  [[nodiscard]] std::size_t size() const noexcept { return graph.size(); }
};

[[nodiscard]] RunBindings
build_run_bindings(const std::shared_ptr<JobState> &state,
                   const std::span<rund::AccelRunBinding> bindings) {
  RunBindings result{};
  if (state == nullptr || state->program == nullptr ||
      state->program->accel == nullptr || state->inputs.empty() ||
      state->outputs.empty()) {
    result.status = Status::fail(Reason::AccelProgramInvalid);
    return result;
  }
  if (std::any_of(state->inputs.begin(), state->inputs.end(),
                  [](const auto &buffer) { return buffer == nullptr; })) {
    result.status = Status::fail(Reason::AccelProgramInvalid);
    return result;
  }
  if (state->program->graph_bindings.empty()) {
    result.status = Status::fail(Reason::RunCapacity);
    return result;
  }
  if (bindings.size() != state->program->graph_bindings.size()) {
    result.status = Status::fail(Reason::RunCapacity);
    return result;
  }
  for (std::size_t index = 0u; index < state->program->graph_bindings.size();
       ++index) {
    const GraphRunBinding &binding = state->program->graph_bindings[index];
    BufferState *const buffer =
        graph_binding_buffer(*state->program, binding, state->inputs,
                             state->outputs, job_graph_buffers(*state));
    if (buffer == nullptr) {
      return result;
    }
    AccelBufferState *const resident = accel_buffer(*buffer);
    if (resident == nullptr) {
      return result;
    }
    const JobBufferView view = binding_view(*state, binding, *buffer);
    bindings[index] = rund::AccelRunBinding{
        .buffer = &resident->buffer,
        .role = binding.role,
        .offset_bytes = view.offset * view.element_bytes,
        .element_count = view.count,
        .stride_bytes = view.stride * view.element_bytes,
        .element_bytes = view.element_bytes,
        .alignment = view.alignment,
    };
  }
  result.graph = bindings;
  result.status = Status::success();
  return result;
}

[[nodiscard]] Result<RunState>
finish_accel(const std::shared_ptr<JobState> &state,
             const rund::AccelEvidence &evidence) {
  const Stats stats =
      stats_from_evidence(state->program->device->backend, evidence,
                          state->program->graph_info.read_bytes);
  if (!evidence.ok) {
    {
      std::lock_guard lock{state->gate};
      if (state->terminal != nullptr) {
        state->terminal->failed_stats = stats;
      }
    }
    return Result<RunState>::fail(
        project_reason(evidence.reason, Reason::BackendFailed));
  }
  RunState run{};
  run.program = state->program;
  std::copy(state->outputs.begin(), state->outputs.end(), run.outputs.begin());
  run.stats = stats;
  return Result<RunState>::success(std::move(run));
}

void CompleteAccel(void *const raw,
                   const rund::AccelEvidence &evidence) noexcept {
  auto *const run = static_cast<AccelRunSlot *>(raw);
  if (run == nullptr || run->completion == nullptr || run->state == nullptr) {
    return;
  }
  std::shared_ptr<JobState> state = run->state;
  std::shared_ptr<void> lifetime = std::move(run->lifetime);
  const JobCompletion completion = run->completion;
  void *const user = run->user;
  run->state.reset();
  run->completion = nullptr;
  run->user = nullptr;
  completion(user, finish_accel(state, evidence));
  (void)lifetime;
}

} // namespace

Result<RunState> finish_job_accel(const std::shared_ptr<JobState> &state,
                                  const rund::AccelEvidence &evidence) {
  return finish_accel(state, evidence);
}

Status
prepare_job_accel(const std::shared_ptr<JobState> &state,
                  const std::span<rund::AccelRunBinding> binding_storage) {
  if (state == nullptr || state->program == nullptr ||
      state->program->device == nullptr) {
    return Status::fail(Reason::ProgramInvalid);
  }
  if (state->program->device->backend == Backend::Cpu) {
    return Status::success();
  }
  if (state->program->empty()) {
    return Status::success();
  }
  AccelDeviceState *const accel = accel_device(*state->program->device);
  if (accel == nullptr || state->program->accel == nullptr) {
    return Status::fail(Reason::AccelProgramInvalid);
  }
  const RunBindings bindings = build_run_bindings(state, binding_storage);
  if (!bindings) {
    return bindings.status;
  }
  const Status arena_ready = prepare_arena(state);
  if (!arena_ready) {
    return arena_ready;
  }
  const JobArena *const arena =
      state->workspace == nullptr || state->workspace->arena == nullptr
          ? nullptr
          : state->workspace->arena.get();
  state->prepared = node::accel::detail::PrepareKernelRun(
      accel->context, state->program->accel->kernel,
      rund::AccelRun{
          .bindings = bindings.data(),
          .binding_count = bindings.size(),
          .tile_count = state->program->count,
          .fresh_evidence = false,
      },
      state->terminal == nullptr
          ? node::accel::detail::KernelPreparationMode::PipelinePrivate
          : node::accel::detail::KernelPreparationMode::Standalone,
      arena == nullptr ? nullptr : &state->views,
      arena == nullptr ? nullptr : &arena->binds,
      arena == nullptr ? nullptr : &arena->scratch);
  return state->prepared.ok
             ? Status::success()
             : Status::fail(project_reason(state->prepared.reason,
                                           Reason::LoweringInvalid));
}

Result<RunState> run_job_accel(const std::shared_ptr<JobState> &state) {
  if (state == nullptr || state->program == nullptr ||
      state->program->device == nullptr || state->program->accel == nullptr ||
      state->outputs.empty() || !state->prepared.ok) {
    return Result<RunState>::fail(Reason::RunInvalid);
  }
  AccelDeviceState *const accel = accel_device(*state->program->device);
  if (accel == nullptr) {
    return Result<RunState>::fail(Reason::AccelProgramInvalid);
  }
  return finish_job_accel(state, node::accel::detail::RunPreparedKernel(
                                     accel->context, state->prepared));
}

Status submit_job_accel(const std::shared_ptr<JobState> &state,
                        std::shared_ptr<void> lifetime,
                        const JobCompletion completion,
                        void *const user) noexcept {
  if (state == nullptr || state->program == nullptr ||
      state->program->device == nullptr || state->program->accel == nullptr ||
      state->outputs.empty() || !state->prepared.ok || completion == nullptr) {
    return Status::fail(Reason::RunInvalid);
  }
  AccelDeviceState *const accel = accel_device(*state->program->device);
  if (accel == nullptr) {
    return Status::fail(Reason::AccelProgramInvalid);
  }
  AccelRunSlot &run = state->accel_run;
  if (run.completion != nullptr) {
    return Status::fail(Reason::JobBusy);
  }
  run.state = state;
  run.lifetime = std::move(lifetime);
  run.completion = completion;
  run.user = user;
  const rund::AccelCheck submitted = node::accel::detail::SubmitPreparedKernel(
      accel->context, state->prepared, run.lifetime, CompleteAccel, &run);
  if (submitted.ok) {
    return Status::success();
  }
  run.state.reset();
  run.lifetime.reset();
  run.completion = nullptr;
  run.user = nullptr;
  return Status::fail(project_reason(submitted.reason, Reason::BackendFailed));
}

} // namespace rund::compute::detail
