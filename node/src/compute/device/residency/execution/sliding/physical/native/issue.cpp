#include "../../internal.hpp"

namespace rund::compute::detail::residency {

bool SlidingOwner::issue_execution_sliding_native(
    const execution::Plan &plan, execution::Sliding &sliding,
    const execution::SlidingProjection &projection,
    const std::span<PageUse> uses, execution::SlidingNative &ticket) noexcept {
  if (ticket || sliding.state_ == nullptr) {
    return false;
  }
  std::lock_guard authority_lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked()) {
    return false;
  }
  execution::Sliding::State &state = *sliding.state_;
  std::lock_guard state_lock{state.gate};
  execution::SlidingProjection expected{};
  execution::Node dispatch{};
  if (!authority_.execution_state_.slot.sliding_admitted ||
      authority_.execution_state_.slot.window_final ||
      authority_.execution_state_.slot.token != state.token ||
      authority_.execution_state_.slot.generation != state.generation ||
      authority_.execution_state_.slot.native_inflight != state.owner ||
      !state.authority_bound || state.model_only || state.closed ||
      state.finalizing ||
      state.invocation.topology() != execution::SlidingTopology::Direct ||
      state.invocation.direct_ == nullptr ||
      state.invocation.direct_->identity() != plan.identity() ||
      plan.identity() != authority_.execution_state_.slot.plan ||
      !state.accepts(projection.coordinate) ||
      projection.coordinate.ordinal != state.admitted ||
      !state.exact(projection, uses, expected) ||
      !plan.project(execution::NodeId{.epoch = projection.coordinate.ordinal,
                                      .phase = execution::Phase::Dispatch},
                    dispatch)) {
    return false;
  }
  execution::Sliding::State::NativeCell &native =
      state.native_cells[static_cast<std::size_t>(
          projection.coordinate.ordinal % execution::SlidingNativeCapacity)];
  if (native.state != execution::NativeState::Ready ||
      native.coordinate != projection.coordinate || native.physical_handoff ||
      native.input_count != dispatch.input_count ||
      native.output_count != dispatch.output_count) {
    return false;
  }
  for (std::size_t local = 0u; local < native.input_count; ++local) {
    const std::uint32_t frame = native.device_input_frames[local];
    if (frame >= authority_.frames_.size() ||
        !authority_.frames_[frame].assigned ||
        authority_.frames_[frame].tier != FrameTier::Device ||
        authority_.frames_[frame].role != FrameRole::Input ||
        authority_.frames_[frame].state != FrameState::Resident ||
        authority_.frames_[frame].key != dispatch.input[local].key) {
      return false;
    }
  }
  for (std::size_t local = 0u; local < native.output_count; ++local) {
    const std::uint32_t frame = native.device_output_frames[local];
    if (frame >= authority_.frames_.size() ||
        !authority_.frames_[frame].assigned ||
        authority_.frames_[frame].tier != FrameTier::Device ||
        authority_.frames_[frame].role != FrameRole::Output ||
        !authority_.frames_[frame].dirty.empty() ||
        (authority_.frames_[frame].state != FrameState::Empty &&
         authority_.frames_[frame].state != FrameState::Resident) ||
        (authority_.frames_[frame].state == FrameState::Resident &&
         authority_.frames_[frame].retain_until != NeverUse &&
         authority_.frames_[frame].retain_until >=
             projection.coordinate.ordinal)) {
      return false;
    }
  }
  const auto map = [](registry_model::Frame &frame, const CacheKey key,
                      const std::uint64_t next_use,
                      const std::uint64_t retain_until) noexcept {
    const registry_model::Frame prior = frame;
    frame = registry_model::Frame{.key = key,
                                  .next_use = next_use,
                                  .retain_until = retain_until,
                                  .state = FrameState::Mapping,
                                  .tier = prior.tier,
                                  .role = prior.role,
                                  .extent = prior.extent,
                                  .view = prior.view,
                                  .assigned = prior.assigned};
  };
  for (std::size_t local = 0u; local < native.output_count; ++local) {
    map(authority_.frames_[native.device_output_frames[local]],
        dispatch.output[local].key, dispatch.output[local].next_use,
        dispatch.output[local].retain_until);
  }
  native.state = execution::NativeState::Submitted;
  native.completion = Status::success();
  native.completion_terminal = execution::TerminalKind::Known;
  native.completion_may_write = false;
  native.physical_handoff = true;
  ++state.admitted;
  ticket.ticket_ = execution::SlidingTicket{
      .plan = state.plan,
      .coordinate = projection.coordinate,
      .token = state.token,
      .generation = state.generation,
      .owner = state.owner,
      .turn = native.turn,
      .expected_bytes = native.promote_bytes,
      .slot = static_cast<std::uint32_t>(projection.coordinate.ordinal %
                                         execution::SlidingNativeCapacity),
      .kind = execution::SlidingTicketKind::Native,
  };
  ticket.input_frames_ = native.device_input_frames;
  ticket.output_frames_ = native.device_output_frames;
  ticket.nonce_ = native.turn;
  ticket.active_mask_ = dispatch.active_mask;
  ticket.input_count_ = native.input_count;
  ticket.output_count_ = native.output_count;
  return true;
}

} // namespace rund::compute::detail::residency
