#include "internal.hpp"

namespace rund::node::accel::detail::prepared::sliding {

void fail_projection(State &state, Slot &slot,
                     const rund::AccelCheck failure) noexcept {
  record_failure(state, slot.coordinate, failure);
  slot.phase = SlotPhase::Done;
}

} // namespace rund::node::accel::detail::prepared::sliding
