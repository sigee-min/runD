#include "internal.hpp"

namespace rund::node::accel::detail::prepared::sliding {

bool service_returned(const std::shared_ptr<State> &state,
                      const std::size_t index) noexcept {
  PreparedResidencySlidingReleaseReturned returned = nullptr;
  void *user = nullptr;
  std::uint64_t coordinate = 0u;
  std::uint64_t turn = 0u;
  std::uint8_t slot_index = 0u;
  {
    std::lock_guard lock{state->gate};
    Slot &slot = state->slots[index];
    if (!state->active || state->final_sent ||
        slot.phase != SlotPhase::Returning) {
      return false;
    }
    returned = state->request.returned;
    user = state->request.user;
    coordinate = slot.coordinate;
    turn = slot.turn;
    slot_index = slot.slot;
    ++state->external_calls;
  }

  const bool reusable = returned(user, coordinate, turn, slot_index);
  bool progressed = false;
  {
    std::lock_guard lock{state->gate};
    if (state->external_calls != 0u) {
      --state->external_calls;
    }
    Slot &slot = state->slots[index];
    if (!state->active || state->final_sent ||
        slot.phase != SlotPhase::Returning || slot.coordinate != coordinate ||
        slot.turn != turn || slot.slot != slot_index || !reusable) {
      return false;
    }
    ++state->released;
    if (state->failed ||
        coordinate >
            std::numeric_limits<std::uint64_t>::max() - state->role_count ||
        coordinate + state->role_count >= state->request.coordinate_count) {
      slot.phase = SlotPhase::Done;
    } else {
      slot.coordinate += state->role_count;
      ++slot.turn;
      slot.selection = {};
      slot.phase = SlotPhase::Idle;
      state->pump_pending = true;
    }
    state->pump_pending =
        state->pump_pending ||
        std::any_of(state->slots.begin(),
                    state->slots.begin() + state->role_count,
                    [](const Slot &candidate) {
                      return candidate.phase == SlotPhase::Returning;
                    });
    progressed = true;
  }
  return progressed;
}

} // namespace rund::node::accel::detail::prepared::sliding
