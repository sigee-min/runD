#include "../../internal.hpp"

namespace rund::compute::detail::residency {

bool SlidingOwner::issue_execution_sliding_persist(
    const execution::Plan &plan, execution::Sliding &sliding,
    const execution::SlidingProjection &projection,
    const std::span<PageUse> uses, const std::size_t use,
    execution::SlidingPersist &ticket) noexcept {
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
  if (local >= output_node.output_count) {
    return false;
  }
  const std::span<execution::Sliding::State::OutputCell> ring =
      state.output_ring(projection.coordinate);
  const auto found =
      std::find_if(ring.begin(), ring.end(),
                   [&](const execution::Sliding::State::OutputCell &cell) {
                     return cell.state == execution::OutputState::Ready &&
                            cell.coordinate == projection.coordinate &&
                            cell.use == use;
                   });
  if (found == ring.end() || found->physical_handoff ||
      found->physical_frame >= authority_.frames_.size()) {
    return false;
  }
  registry_model::Frame &host = authority_.frames_[found->physical_frame];
  if (host.state != FrameState::Dirty ||
      host.key != output_node.output[local].key ||
      host.dirty != output_node.output[local].dirty) {
    return false;
  }
  const auto earlier = [&](const execution::Sliding::State::OutputCell &cell) {
    if (cell.state == execution::OutputState::Free || &cell == &*found) {
      return false;
    }
    return cell.coordinate.ordinal < found->coordinate.ordinal ||
           (cell.coordinate.ordinal == found->coordinate.ordinal &&
            cell.use < found->use);
  };
  if (std::any_of(state.outputs.begin(),
                  state.outputs.begin() + state.output_count, earlier)) {
    return false;
  }
  found->state = execution::OutputState::Persisting;
  found->persist_sequence = state.persist_issued++;
  found->completion = Status::success();
  found->completion_terminal = execution::TerminalKind::Known;
  found->completion_bytes = 0u;
  found->completion_may_write = false;
  found->physical_handoff = true;
  ++state.persist_calls;
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
      .kind = execution::SlidingTicketKind::Persist,
  };
  ticket.key_ = host.key;
  ticket.extent_ = host.dirty;
  ticket.frame_ = found->physical_frame;
  ticket.nonce_ = found->turn;
  return true;
}

} // namespace rund::compute::detail::residency
