#include "internal.hpp"

namespace rund::node::accel::detail::prepared::sliding {

bool begin_projection(State &state, const std::size_t index,
                      ProjectHandoff &handoff) noexcept {
  std::lock_guard lock{state.gate};
  Slot &slot = state.slots[index];
  if (!state.active || state.failed || slot.phase != SlotPhase::Idle) {
    return false;
  }
  slot.phase = SlotPhase::Projecting;
  handoff = ProjectHandoff{.callback = state.request.project,
                           .user = state.request.user,
                           .coordinate = slot.coordinate,
                           .turn = slot.turn,
                           .slot = slot.slot};
  ++state.external_calls;
  return true;
}

bool complete_projection(
    State &state, const std::size_t index, const ProjectHandoff &handoff,
    const PreparedResidencySlidingProjection projected,
    const PreparedResidencySlidingSelection &selection) noexcept {
  std::lock_guard lock{state.gate};
  Slot &slot = state.slots[index];
  if (state.external_calls != 0u) {
    --state.external_calls;
  }
  if (!state.active || slot.phase != SlotPhase::Projecting ||
      slot.coordinate != handoff.coordinate || slot.turn != handoff.turn) {
    return false;
  }
  if (state.failed) {
    slot.phase = SlotPhase::Done;
    return false;
  }
  if (projected == PreparedResidencySlidingProjection::Pending) {
    slot.phase = SlotPhase::Idle;
    return false;
  }
  if (projected != PreparedResidencySlidingProjection::Ready ||
      !valid_selection(selection)) {
    fail_projection(state, slot, {false, "accel_kernel_pipeline_invalid"});
    return false;
  }
  const PreparedResidencySlidingRole &role = state.roles[index];
  const bool generation_valid =
      role.first_control_generation != 0u &&
      role.control_generation_stride != 0u &&
      handoff.turn <= (std::numeric_limits<std::uint32_t>::max() -
                       role.first_control_generation) /
                          role.control_generation_stride &&
      selection.control_generation ==
          role.first_control_generation +
              static_cast<std::uint32_t>(handoff.turn) *
                  role.control_generation_stride &&
      handoff.turn != std::numeric_limits<std::uint64_t>::max() &&
      selection.descriptor_generation == handoff.turn + 1u;
  if (!generation_valid) {
    fail_projection(state, slot, {false, "accel_kernel_pipeline_invalid"});
    return false;
  }
  slot.selection = selection;
  slot.inline_terminal = false;
  slot.same_thread_inline = false;
  slot.submit_returned.store(false, std::memory_order_relaxed);
  slot.phase = SlotPhase::Submitting;
  return true;
}

} // namespace rund::node::accel::detail::prepared::sliding
