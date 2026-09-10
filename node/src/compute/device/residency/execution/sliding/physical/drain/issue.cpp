#include "../../internal.hpp"

namespace rund::compute::detail::residency {

bool SlidingOwner::issue_execution_sliding_drain(
    const execution::Plan &plan, execution::Sliding &sliding,
    const execution::SlidingProjection &projection,
    const std::span<PageUse> uses, const std::size_t use,
    execution::SlidingDrain &ticket) noexcept {
  if (ticket || sliding.state_ == nullptr ||
      use > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  std::lock_guard authority_lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked()) {
    return false;
  }
  execution::Sliding::State &state = *sliding.state_;
  std::lock_guard state_lock{state.gate};
  execution::SlidingProjection expected{};
  execution::Node output_node{};
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
      !state.exact(projection, uses, expected) ||
      !state.invocation.persist_use(projection, uses, use) ||
      use < projection.fetch_count ||
      !plan.project(execution::NodeId{.epoch = projection.coordinate.ordinal,
                                      .phase = execution::Phase::Output},
                    output_node)) {
    return false;
  }
  const std::size_t local = use - projection.fetch_count;
  execution::Sliding::State::NativeCell &native =
      state.native_cells[static_cast<std::size_t>(
          projection.coordinate.ordinal % execution::SlidingNativeCapacity)];
  if (local >= native.output_count || local >= output_node.output_count ||
      native.coordinate != projection.coordinate ||
      native.state != execution::NativeState::Draining ||
      native.physical_handoff) {
    return false;
  }
  const std::span<execution::Sliding::State::OutputCell> ring =
      state.output_ring(projection.coordinate);
  const auto found =
      std::find_if(ring.begin(), ring.end(),
                   [&](const execution::Sliding::State::OutputCell &cell) {
                     return cell.state == execution::OutputState::Reserved &&
                            cell.coordinate == projection.coordinate &&
                            cell.use == use;
                   });
  if (found == ring.end() || found->physical_handoff ||
      found->expected_bytes != output_node.output[local].dirty.bytes) {
    return false;
  }
  const std::uint32_t device_frame = native.device_output_frames[local];
  const std::uint32_t host_frame = found->physical_frame;
  if (device_frame >= authority_.frames_.size() ||
      host_frame >= authority_.frames_.size()) {
    return false;
  }
  registry_model::Frame &device = authority_.frames_[device_frame];
  registry_model::Frame &host = authority_.frames_[host_frame];
  if (!device.assigned || device.tier != FrameTier::Device ||
      device.role != FrameRole::Output || device.state != FrameState::Dirty ||
      device.key != output_node.output[local].key ||
      device.dirty != output_node.output[local].dirty || !host.assigned ||
      host.tier != FrameTier::Host || host.role != FrameRole::Output ||
      !host.dirty.empty() ||
      (host.state != FrameState::Empty && host.state != FrameState::Resident)) {
    return false;
  }
  const registry_model::Frame prior = host;
  host = registry_model::Frame{.key = output_node.output[local].key,
                               .next_use = output_node.output[local].next_use,
                               .retain_until =
                                   output_node.output[local].retain_until,
                               .state = FrameState::Mapping,
                               .tier = prior.tier,
                               .role = prior.role,
                               .extent = prior.extent,
                               .view = prior.view,
                               .assigned = prior.assigned};
  found->state = execution::OutputState::Draining;
  found->completion = Status::success();
  found->completion_terminal = execution::TerminalKind::Known;
  found->completion_bytes = 0u;
  found->completion_may_write = false;
  found->physical_handoff = true;
  ++state.drain_calls;
  ticket.ticket_ = execution::SlidingTicket{
      .plan = state.plan,
      .coordinate = projection.coordinate,
      .token = state.token,
      .generation = state.generation,
      .owner = state.owner,
      .turn = found->turn,
      .expected_bytes = found->expected_bytes,
      .slot = static_cast<std::uint32_t>(found - ring.begin()),
      .use = found->use,
      .kind = execution::SlidingTicketKind::Drain,
  };
  ticket.device_frame_ = device_frame;
  ticket.host_frame_ = host_frame;
  ticket.nonce_ = found->turn;
  return true;
}

} // namespace rund::compute::detail::residency
