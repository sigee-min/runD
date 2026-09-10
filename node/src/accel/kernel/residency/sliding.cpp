#include "sliding/internal.hpp"

#include <new>

namespace rund::node::accel::detail {

using namespace prepared::sliding;

rund::AccelCheck PrepareKernelPipelineSliding(
    const rund::AccelContext &context,
    const std::span<const PreparedResidencySlidingRole> roles,
    const ResidencySlidingMemory memory,
    PreparedResidencySlidingControl &control) noexcept {
  if (roles.empty() || roles.size() > ResidencySlidingCapacity ||
      control.state != nullptr) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  Admission admission{};
  const rund::AccelCheck admitted = admit(context, roles, memory, admission);
  if (!admitted.ok) {
    return admitted;
  }
  std::shared_ptr<State> state;
  try {
    state = std::make_shared<State>();
    state->service = std::make_shared<Service>();
  } catch (const std::bad_alloc &) {
    return {false, "compute_pipeline_memory_budget"};
  }
  if (!state->service->Start(service_pump, service_finish)) {
    return {false, "compute_pipeline_memory_budget"};
  }
  state->context = context;
  state->role_count = admission.role_count;
  for (std::size_t slot = 0u; slot < roles.size(); ++slot) {
    const PreparedResidencySlidingRole &role = roles[slot];
    state->roles[slot] = role;
    state->pipelines[slot] = admission.pipelines[slot];
    state->slots[slot].state = state.get();
    state->slots[slot].pipeline = role.pipeline;
    state->slots[slot].slot = static_cast<std::uint8_t>(slot);
  }
  state->capability = admission.capability;
  control.capability = admission.capability;
  control.state = std::move(state);
  return {true, "ok"};
}

rund::AccelCheck SubmitPreparedKernelPipelineSliding(
    const rund::AccelContext &context,
    const PreparedResidencySlidingRequest &request,
    PreparedResidencySlidingControl &control) noexcept {
  const std::shared_ptr<State> state = control.state;
  if (state == nullptr || request.plan_identity == 0u || request.token == 0u ||
      request.generation == 0u || request.coordinate_count == 0u ||
      request.project == nullptr || request.release == nullptr ||
      request.returned == nullptr || request.final == nullptr ||
      request.user == nullptr || request.memory != control.capability.memory ||
      !same_roles(*state, request)) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  std::array<std::unique_lock<std::mutex>, ResidencySlidingCapacity> claims{};
  for (std::size_t slot = 0u; slot < state->role_count; ++slot) {
    claims[slot] = std::unique_lock<std::mutex>{
        state->pipelines[slot]->submission.mutex, std::try_to_lock};
    if (!claims[slot].owns_lock()) {
      return {false, "compute_pipeline_busy"};
    }
  }
  {
    std::lock_guard lock{state->gate};
    if (state->active || state->unknown || state->quarantine != nullptr ||
        state->context.id != context.id ||
        state->context.owner != context.owner) {
      return {false, "compute_pipeline_busy"};
    }
    for (std::size_t slot = 0u; slot < state->role_count; ++slot) {
      prepared::PipelineSubmission &submission =
          state->pipelines[slot]->submission;
      if (submission.active() || submission.quarantined) {
        return {false, "compute_pipeline_busy"};
      }
    }
    state->request = request;
    state->accepted = 0u;
    state->released = 0u;
    state->queue_calls = 0u;
    state->inflight = 0u;
    state->inflight_peak = 0u;
    state->external_calls = 0u;
    state->first_failure = std::numeric_limits<std::uint64_t>::max();
    state->failure = {true, "ok"};
    state->failed = false;
    state->unknown = false;
    state->final_sent = false;
    state->pumping = false;
    state->pump_pending = false;
    state->active = true;
    state->active_owner = state;
    for (std::size_t slot = 0u; slot < state->role_count; ++slot) {
      Slot &cell = state->slots[slot];
      cell.coordinate = slot;
      cell.turn = 0u;
      cell.selection = {};
      cell.inline_terminal = false;
      cell.phase =
          slot < request.coordinate_count ? SlotPhase::Idle : SlotPhase::Done;
      state->pipelines[slot]->submission.sliding = state.get();
    }
  }
  claims = {};
  pump(state);
  return {true, "ok"};
}

rund::AccelCheck WakePreparedKernelPipelineSliding(
    PreparedResidencySlidingControl &control) noexcept {
  const std::shared_ptr<State> state = control.state;
  if (state == nullptr) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  {
    std::lock_guard lock{state->gate};
    if (!state->active || state->final_sent || state->failed) {
      return {false, "accel_kernel_pipeline_invalid"};
    }
  }
  pump(state);
  return {true, "ok"};
}

} // namespace rund::node::accel::detail
