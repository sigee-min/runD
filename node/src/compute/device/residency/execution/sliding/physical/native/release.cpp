#include "../../internal.hpp"

namespace rund::compute::detail::residency {

bool SlidingOwner::release_execution_sliding_native(
    execution::Sliding &sliding, execution::SlidingNative &&ticket) noexcept {
  if (sliding.state_ == nullptr || !ticket) {
    return false;
  }
  std::lock_guard authority_lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked()) {
    return false;
  }
  execution::Sliding::State &state = *sliding.state_;
  std::lock_guard state_lock{state.gate};
  execution::Sliding::State::NativeCell *const native =
      state.native(ticket.ticket_, execution::SlidingTicketKind::Native);
  if (!authority_.execution_state_.slot.sliding_admitted ||
      authority_.execution_state_.slot.token != state.token ||
      authority_.execution_state_.slot.generation != state.generation ||
      authority_.execution_state_.slot.native_inflight != state.owner ||
      !state.authority_bound || state.model_only || state.closed ||
      state.finalizing || native == nullptr ||
      native->state != execution::NativeState::Completing ||
      !native->physical_handoff || ticket.nonce_ != native->turn ||
      ticket.input_frames_ != native->device_input_frames ||
      ticket.output_frames_ != native->device_output_frames) {
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
  const bool known =
      native->completion_terminal == execution::TerminalKind::Known;
  const bool success =
      known && native->completion && state.accepts(native->coordinate);
  if (success) {
    execution::Node dispatch{};
    if (!state.invocation.direct_->project(
            execution::NodeId{.epoch = native->coordinate.ordinal,
                              .phase = execution::Phase::Dispatch},
            dispatch)) {
      return false;
    }
    for (std::size_t local = 0u; local < native->output_count; ++local) {
      registry_model::Frame &frame =
          authority_.frames_[native->device_output_frames[local]];
      if (frame.state != FrameState::Mapping) {
        return false;
      }
      frame.state = FrameState::Dirty;
      frame.dirty = dispatch.output[local].dirty;
    }
    for (execution::Sliding::State::InputCell &input : state.inputs) {
      if (input.state == execution::InputState::Promoting &&
          input.coordinate == native->coordinate) {
        input.state = execution::InputState::Ready;
      }
    }
    native->state = execution::NativeState::Draining;
  } else if (!known) {
    native->state = execution::NativeState::Quarantined;
    for (execution::Sliding::State::InputCell &input : state.inputs) {
      if (input.state == execution::InputState::Promoting &&
          input.coordinate == native->coordinate) {
        input.state = execution::InputState::Quarantined;
      }
    }
    for (execution::Sliding::State::OutputCell &output : state.outputs) {
      if (output.state == execution::OutputState::Reserved &&
          output.coordinate == native->coordinate) {
        output.state = execution::OutputState::Quarantined;
      }
    }
  } else {
    for (std::size_t local = 0u; local < native->output_count; ++local) {
      clear(authority_.frames_[native->device_output_frames[local]]);
    }
    for (execution::Sliding::State::InputCell &input : state.inputs) {
      if (input.state == execution::InputState::Promoting &&
          input.coordinate == native->coordinate) {
        input.state = state.accepts(input.coordinate)
                          ? execution::InputState::Ready
                          : execution::InputState::Free;
      }
    }
    for (execution::Sliding::State::OutputCell &output : state.outputs) {
      if (output.state == execution::OutputState::Reserved &&
          output.coordinate == native->coordinate) {
        output.state = execution::OutputState::Free;
      }
    }
    native->state = execution::NativeState::Free;
    state.retire(native->coordinate, false);
  }
  native->physical_handoff = false;
  ticket.ticket_ = {};
  ticket.input_frames_ = {};
  ticket.output_frames_ = {};
  ticket.nonce_ = 0u;
  ticket.active_mask_ = 0u;
  ticket.input_count_ = 0u;
  ticket.output_count_ = 0u;
  return true;
}

} // namespace rund::compute::detail::residency
