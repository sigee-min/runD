#include "../../internal.hpp"

namespace rund::compute::detail::residency {

bool SlidingOwner::release_execution_sliding_promote(
    execution::Sliding &sliding, execution::SlidingPromote &&ticket) noexcept {
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
      state.native(ticket.ticket_, execution::SlidingTicketKind::Promote);
  if (!authority_.execution_state_.slot.sliding_admitted ||
      authority_.execution_state_.slot.token != state.token ||
      authority_.execution_state_.slot.generation != state.generation ||
      authority_.execution_state_.slot.native_inflight != state.owner ||
      !state.authority_bound || state.model_only || state.closed ||
      state.finalizing || native == nullptr ||
      native->state != execution::NativeState::Completing ||
      !native->physical_handoff || ticket.nonce_ != native->turn ||
      ticket.source_count_ != native->source_count ||
      ticket.input_count_ != native->input_count ||
      ticket.output_count_ != native->output_count ||
      ticket.transfer_mask_ != native->transfer_mask ||
      ticket.host_frames_ != native->host_frames ||
      ticket.device_input_frames_ != native->device_input_frames ||
      ticket.device_output_frames_ != native->device_output_frames) {
    return false;
  }
  const auto clear_frame = [](registry_model::Frame &frame) noexcept {
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
    for (std::size_t local = 0u; local < native->input_count; ++local) {
      if ((native->transfer_mask & (std::uint32_t{1u} << local)) != 0u) {
        registry_model::Frame &device =
            authority_.frames_[native->device_input_frames[local]];
        if (device.state != FrameState::Mapping) {
          return false;
        }
        device.state = FrameState::Resident;
      }
    }
    native->state = execution::NativeState::Ready;
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
    const bool wrote = native->completion_may_write ||
                       (native->completion && native->transfer_mask != 0u);
    if (native->completion && !state.accepts(native->coordinate) && wrote) {
      state.fail(native->coordinate, Status::fail(Reason::CompletionInvalid),
                 execution::TerminalKind::Known, true);
    }
    for (execution::Sliding::State::InputCell &input : state.inputs) {
      if (input.state == execution::InputState::Promoting &&
          input.coordinate == native->coordinate) {
        input.state = execution::InputState::Free;
      }
    }
    for (std::size_t local = 0u; local < native->output_count; ++local) {
      clear_frame(authority_.frames_[native->device_output_frames[local]]);
      clear_frame(authority_.frames_[ticket.host_output_frames_[local]]);
    }
    for (execution::Sliding::State::OutputCell &output : state.outputs) {
      if (output.state == execution::OutputState::Reserved &&
          output.coordinate == native->coordinate) {
        output.state = execution::OutputState::Free;
      }
    }
    for (std::size_t local = 0u; local < native->input_count; ++local) {
      if ((native->transfer_mask & (std::uint32_t{1u} << local)) != 0u) {
        // Authority owns this exact Mapping target.  A Known unsuccessful or
        // suppressed Promote invalidates it here before admitting any reuse;
        // no controller-authored invalidation receipt can bypass this step.
        clear_frame(authority_.frames_[native->device_input_frames[local]]);
      }
    }
    native->state = execution::NativeState::Free;
  }
  native->physical_handoff = false;
  ticket.ticket_ = {};
  ticket.host_frames_ = {};
  ticket.device_input_frames_ = {};
  ticket.device_output_frames_ = {};
  ticket.host_output_frames_ = {};
  ticket.targets_ = {};
  ticket.slices_ = {};
  ticket.nonce_ = 0u;
  ticket.transfer_mask_ = 0u;
  ticket.source_count_ = 0u;
  ticket.input_count_ = 0u;
  ticket.output_count_ = 0u;
  ticket.slice_count_ = 0u;
  return true;
}

} // namespace rund::compute::detail::residency
