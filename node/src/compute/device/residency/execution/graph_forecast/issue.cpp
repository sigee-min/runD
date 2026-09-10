#include "internal.hpp"

namespace rund::compute::detail::residency {

AuthorityResult GraphForecastOwner::issue_graph_forecast(
    std::shared_ptr<const ResidencyPlan> owner,
    const TiledGraphInvocation &invocation, const std::uint64_t batch,
    const std::size_t stage, const std::uint32_t resource,
    const std::span<const PageUse> selected,
    const GraphMaterialization materialization, const FrameRegion region,
    execution::GraphForecast &ticket) noexcept {
  if (ticket) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }
  graph_forecast_detail::Projection projected{};
  if (!graph_forecast_detail::validate_projection(
          owner, invocation, batch, stage, resource, selected, materialization,
          region, projected)) {
    return AuthorityResult{.failure = AuthorityFailure::Invalid};
  }
  std::array<execution::GraphForecastPage, execution::GraphForecastCapacity>
      pages{};
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() ||
      authority_.view_commit_inflight_locked()) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }
  if (!authority_.graph_forecast_quarantine_slot_locked()) {
    return AuthorityResult{.failure = AuthorityFailure::Capacity};
  }
  const AuthorityResult acquired = authority_.begin_graph_locked(
      selected, materialization, region, projected.epoch.ordinal);
  if (!acquired) {
    return acquired;
  }
  const auto fail = [&]() noexcept {
    const bool closed =
        authority_.complete_locked(acquired.lease.token, false, true);
    return AuthorityResult{.failure = closed ? AuthorityFailure::Invalid
                                             : AuthorityFailure::Busy};
  };
  if (acquired.lease.bindings.size() != selected.size() ||
      std::any_of(acquired.lease.transitions.begin(),
                  acquired.lease.transitions.end(),
                  [](const CacheTransition transition) {
                    return transition.kind == TransitionKind::Writeback;
                  })) {
    return fail();
  }
  for (std::size_t index = 0u; index < selected.size(); ++index) {
    const CacheBinding binding = acquired.lease.bindings[index];
    CacheKey key{};
    std::uint64_t offset = 0u;
    std::uint64_t bytes = 0u;
    if (binding.access != Access::Read ||
        !project_graph_cache_key(materialization, selected[index].key, key) ||
        key != binding.key || binding.frame < region.first ||
        binding.frame - region.first >= region.count ||
        !kernel::checked::mul(selected[index].key.page,
                              materialization.page_bytes, offset) ||
        !invocation.page_bytes(resource, selected[index].key.page, bytes)) {
      return fail();
    }
    pages[index] = execution::GraphForecastPage{
        .use = selected[index].key,
        .key = key,
        .offset = offset,
        .bytes = bytes,
        .frame = binding.frame,
        .fetch = binding.fetch,
    };
  }
  if (!authority_.activate_locked(acquired.lease.token)) {
    return fail();
  }
  ticket.owner_ = &authority_;
  ticket.plan_owner_ = std::move(owner);
  ticket.pages_ = pages;
  ticket.plan_ = ticket.plan_owner_->identity();
  ticket.completion_ = Status::fail(Reason::CompletionInvalid);
  ticket.token_ = acquired.lease.token;
  ticket.generation_ = acquired.lease.generation;
  ticket.coordinate_ = projected.epoch.ordinal;
  ticket.page_count_ = selected.size();
  ticket.terminal_ = execution::TerminalKind::Known;
  ticket.completion_may_write_ = false;
  ticket.terminalled_ = false;
  return AuthorityResult{.failure = AuthorityFailure::None};
}

} // namespace rund::compute::detail::residency
