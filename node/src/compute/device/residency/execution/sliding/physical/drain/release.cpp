#include "../../internal.hpp"

namespace rund::compute::detail::residency {

bool SlidingOwner::release_execution_sliding_drain(
    execution::Sliding &sliding, execution::SlidingDrain &&ticket) noexcept {
  if (sliding.state_ == nullptr || !ticket) {
    return false;
  }
  std::lock_guard authority_lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked()) {
    return false;
  }
  execution::Sliding::State &state = *sliding.state_;
  std::lock_guard state_lock{state.gate};
  execution::Sliding::State::OutputCell *const output =
      state.output(ticket.ticket_, execution::SlidingTicketKind::Drain);
  if (!authority_.execution_state_.slot.sliding_admitted ||
      authority_.execution_state_.slot.token != state.token ||
      authority_.execution_state_.slot.generation != state.generation ||
      authority_.execution_state_.slot.native_inflight != state.owner ||
      !state.authority_bound || state.model_only || state.closed ||
      state.finalizing || output == nullptr ||
      output->state != execution::OutputState::Completing ||
      !output->physical_handoff || ticket.nonce_ != output->turn ||
      ticket.host_frame_ != output->physical_frame ||
      ticket.host_frame_ >= authority_.frames_.size() ||
      ticket.device_frame_ >= authority_.frames_.size()) {
    return false;
  }
  execution::Sliding::State::NativeCell &native =
      state.native_cells[static_cast<std::size_t>(
          ticket.ticket_.coordinate.ordinal %
          execution::SlidingNativeCapacity)];
  if (native.coordinate != ticket.ticket_.coordinate ||
      native.state != execution::NativeState::Draining) {
    return false;
  }
  const auto clear = [](registry_model::Frame &frame) noexcept {
    const registry_model::Frame prior = frame;
    frame = registry_model::Frame{.state = FrameState::Empty,
                                  .tier = prior.tier,
                                  .role = prior.role,
                                  .extent = prior.extent,
                                  .view = prior.view,
                                  .assigned = prior.assigned};
  };
  registry_model::Frame &host = authority_.frames_[ticket.host_frame_];
  registry_model::Frame &device = authority_.frames_[ticket.device_frame_];
  const bool known =
      output->completion_terminal == execution::TerminalKind::Known;
  const bool success = known && output->completion &&
                       state.accepts(output->coordinate) &&
                       !output->invalidate_after_terminal;
  if (success) {
    if (host.state != FrameState::Mapping ||
        device.state != FrameState::Dirty) {
      return false;
    }
    host.state = FrameState::Dirty;
    host.dirty = device.dirty;
    clear(device);
    output->state = execution::OutputState::Ready;
    ++native.resolved;
    state.resolve_native(native);
  } else if (!known) {
    output->state = execution::OutputState::Quarantined;
    native.state = execution::NativeState::Quarantined;
  } else {
    clear(host);
    clear(device);
    output->state = execution::OutputState::Free;
    ++native.resolved;
    for (execution::Sliding::State::OutputCell &sibling : state.outputs) {
      if (&sibling != output && sibling.coordinate == native.coordinate &&
          sibling.state == execution::OutputState::Reserved) {
        clear(authority_.frames_[sibling.physical_frame]);
        sibling.state = execution::OutputState::Free;
        ++native.resolved;
      }
    }
    state.resolve_native(native);
  }
  output->physical_handoff = false;
  ticket.ticket_ = {};
  ticket.device_frame_ = 0u;
  ticket.host_frame_ = 0u;
  ticket.nonce_ = 0u;
  return true;
}

} // namespace rund::compute::detail::residency
