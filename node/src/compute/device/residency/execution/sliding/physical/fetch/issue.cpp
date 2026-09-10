#include "commit.hpp"
#include "credentials.hpp"
#include "projection.hpp"
#include "selection.hpp"
#include "ticket.hpp"

namespace rund::compute::detail::residency {

bool SlidingOwner::issue_execution_sliding_fetch(
    const execution::Plan &plan, execution::Sliding &sliding,
    const execution::SlidingProjection &projection,
    const std::span<PageUse> uses, const std::size_t use,
    execution::SlidingFetch &ticket) noexcept {
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
  const bool invocation_matches =
      state.invocation.topology() == execution::SlidingTopology::Direct &&
      state.invocation.direct_ != nullptr &&
      state.invocation.direct_->identity() == plan.identity();
  if (!physical::fetch::valid_credentials(authority_.execution_state_.slot,
                                          plan, state, projection,
                                          invocation_matches)) {
    return false;
  }
  physical::fetch::Projection validated{};
  if (!physical::fetch::validate_projection(plan, state, projection, uses, use,
                                            authority_.frames_.size(),
                                            validated)) {
    return false;
  }
  physical::fetch::Selection selected{};
  const std::size_t active_banks =
      static_cast<std::size_t>(std::min<std::uint64_t>(
          authority_.execution_state_.slot.epochs, execution::BankCapacity));
  if (!physical::fetch::select_physical(
          state, std::span<const registry_model::Frame>{authority_.frames_},
          projection, validated, active_banks, selected)) {
    return false;
  }
  physical::fetch::Commit committed{};
  if (!physical::fetch::commit_state(
          state, std::span<registry_model::Frame>{authority_.frames_},
          projection, uses, use, validated, selected, committed)) {
    return false;
  }
  if (!selected.hit) {
    clear_sliding_frame_observation(committed.physical_frame);
  }
  physical::fetch::mint_ticket(
      state.plan, state.token, state.generation, state.owner, projection, use,
      validated, selected, committed, ticket.ticket_, ticket.source_,
      ticket.reuse_, ticket.frame_, ticket.reuse_frame_, ticket.backing_bytes_,
      ticket.nonce_, ticket.backing_, ticket.reuses_frame_);
  return true;
}

} // namespace rund::compute::detail::residency
