#include "internal.hpp"

namespace rund::node::accel::detail::prepared::sliding {

void emit_final(const std::shared_ptr<State> &state) noexcept {
  PreparedResidencySlidingFinalCompletion completion = nullptr;
  void *user = nullptr;
  BackendResidencySlidingFinal final{};
  bool known = false;
  {
    std::lock_guard lock{state->gate};
    if (state->final_sent || !state->active) {
      return;
    }
    const bool quiescent =
        state->inflight == 0u && state->external_calls == 0u &&
        !state->pumping &&
        !state->service_queued.load(std::memory_order_acquire) &&
        std::none_of(state->slots.begin(),
                     state->slots.begin() + state->role_count,
                     [](const Slot &slot) {
                       return slot.phase == SlotPhase::Returning;
                     });
    const bool success = !state->failed && !state->unknown &&
                         state->released == state->request.coordinate_count &&
                         quiescent;
    const bool known_failure = state->failed && !state->unknown && quiescent;
    const bool unknown_terminal = state->unknown && quiescent;
    if (!success && !known_failure && !unknown_terminal) {
      return;
    }
    state->active = false;
    state->final_sent = true;
    known = !state->unknown;
    completion = state->request.final;
    user = state->request.user;
    final = BackendResidencySlidingFinal{
        .check = success ? rund::AccelCheck{true, "ok"} : state->failure,
        .terminal =
            known ? NativeTerminal::Known : NativeTerminal::UnknownMayWrite,
        .plan_identity = state->request.plan_identity,
        .token = state->request.token,
        .generation = state->request.generation,
        .coordinate_count = state->request.coordinate_count,
        .accepted_coordinates = state->accepted,
        .released_coordinates = state->released,
        .queue_calls = state->queue_calls,
        .native_inflight_peak = state->inflight_peak,
        .retained_bytes = state->capability.retained_bytes,
        .transient_bytes = state->capability.transient_bytes,
        .first_failure_coordinate =
            state->failed ? state->first_failure
                          : std::numeric_limits<std::uint64_t>::max(),
        .completed_ns = MonotonicNanoseconds(),
    };
    if (!known) {
      state->quarantine = state;
    } else {
      state->request = {};
      state->active_owner.reset();
    }
  }
  state->pump_ready.notify_all();
  if (known) {
    release_claims_known(*state);
  } else {
    for (std::size_t slot = 0u; slot < state->role_count; ++slot) {
      std::lock_guard lock{state->pipelines[slot]->submission.mutex};
      state->pipelines[slot]->submission.quarantined = true;
    }
  }
  completion(user, std::move(final));
}

} // namespace rund::node::accel::detail::prepared::sliding
