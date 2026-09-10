#include "internal.hpp"

namespace rund::node::accel::detail::prepared::sliding {
namespace {

[[nodiscard]] bool acquire_pump(State &state) noexcept {
  std::unique_lock lock{state.gate};
  if (!state.active || state.final_sent) {
    return false;
  }
  if (state.external_calls != 0u) {
    state.pump_pending = true;
    return false;
  }
  if (state.pumping && PumpingState == &state) {
    state.pump_pending = true;
    return false;
  }
  while (state.pumping && state.active && !state.final_sent) {
    state.pump_ready.wait(lock);
  }
  if (!state.active || state.final_sent) {
    return false;
  }
  state.pumping = true;
  state.pump_pending = false;
  return true;
}

[[nodiscard]] bool release_pump(State &state) noexcept {
  std::lock_guard lock{state.gate};
  const bool rescan = state.pump_pending;
  state.pumping = false;
  state.pump_pending = false;
  return rescan;
}

void finish_pump(const std::shared_ptr<State> &state,
                 const bool rescan) noexcept {
  state->pump_ready.notify_all();
  bool terminal_ready = false;
  bool active = false;
  {
    std::lock_guard lock{state->gate};
    terminal_ready =
        state->active && state->inflight == 0u && state->external_calls == 0u &&
        (state->failed || state->released == state->request.coordinate_count);
    active = state->active && !state->final_sent;
  }
  if (terminal_ready) {
    emit_final(state);
  }
  if (rescan && active) {
    schedule_service(*state);
  }
}

} // namespace

thread_local State *PumpingState = nullptr;

void pump(const std::shared_ptr<State> &state) noexcept {
  if (!acquire_pump(*state)) {
    return;
  }
  State *const prior_pumping = PumpingState;
  PumpingState = state.get();
  for (std::size_t index = 0u; index < state->role_count; ++index) {
    if (service_returned(state, index)) {
      continue;
    }
    ProjectHandoff handoff{};
    if (!begin_projection(*state, index, handoff)) {
      continue;
    }
    PreparedResidencySlidingSelection selection{};
    const PreparedResidencySlidingProjection projected =
        handoff.callback(handoff.user, handoff.coordinate, handoff.turn,
                         handoff.slot, selection);
    if (complete_projection(*state, index, handoff, projected, selection)) {
      submit_slot(state, index);
    }
  }
  const bool rescan = release_pump(*state);
  PumpingState = prior_pumping;
  finish_pump(state, rescan);
}

} // namespace rund::node::accel::detail::prepared::sliding
