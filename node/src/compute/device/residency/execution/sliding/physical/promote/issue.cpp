#include "commit.hpp"
#include "credentials.hpp"
#include "projection.hpp"
#include "selection.hpp"
#include "ticket.hpp"

namespace rund::compute::detail::residency {

bool SlidingOwner::issue_execution_sliding_promote(
    const execution::Plan &plan, execution::Sliding &sliding,
    const execution::SlidingProjection &projection,
    const std::span<PageUse> uses, execution::SlidingPromote &ticket) noexcept {
  if (ticket || sliding.state_ == nullptr) {
    return false;
  }
  std::lock_guard authority_lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked()) {
    return false;
  }
  execution::Sliding::State &state = *sliding.state_;
  std::lock_guard state_lock{state.gate};
  const bool invocation_matches =
      state.invocation.topology() == execution::SlidingTopology::Direct &&
      state.invocation.direct_ != nullptr &&
      state.invocation.direct_->identity() == plan.identity();
  if (!physical::promote::valid_credentials(authority_.execution_state_.slot,
                                            plan, state, projection,
                                            invocation_matches)) {
    return false;
  }
  physical::promote::Projection validated{};
  if (!physical::promote::validate_projection(plan, state, projection, uses,
                                              authority_.frames_.size(),
                                              validated)) {
    return false;
  }
  physical::promote::Selection selected{};
  if (!physical::promote::select_physical(
          plan, state,
          std::span<const registry_model::Frame>{authority_.frames_},
          projection, uses, validated, selected)) {
    return false;
  }
  const physical::promote::Commit committed = physical::promote::commit_state(
      state, std::span<registry_model::Frame>{authority_.frames_}, projection,
      validated, selected);
  physical::promote::mint_ticket(
      state.plan, state.token, state.generation, state.owner, projection,
      selected, validated, committed, ticket.ticket_, ticket.host_frames_,
      ticket.device_input_frames_, ticket.device_output_frames_,
      ticket.host_output_frames_, ticket.targets_, ticket.slices_,
      ticket.nonce_, ticket.transfer_mask_, ticket.source_count_,
      ticket.input_count_, ticket.output_count_, ticket.slice_count_);
  return true;
}

} // namespace rund::compute::detail::residency
