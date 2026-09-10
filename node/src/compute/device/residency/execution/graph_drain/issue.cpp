#include "internal.hpp"
#include "../../registry/graph_drain_owner.hpp"

namespace rund::compute::detail::residency {

AuthorityResult GraphDrainOwner::issue_graph_drain(
    std::shared_ptr<const ResidencyPlan> owner,
    const TiledGraphInvocation &invocation, const std::uint64_t batch,
    const std::size_t stage, const std::uint32_t resource,
    const std::span<const PageUse> selected,
    const GraphMaterialization materialization, const FrameRegion source_region,
    const FrameRegion target_region, execution::GraphDrain &ticket) noexcept {
  if (ticket) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }
  graph_drain_detail::Projection projected{};
  if (!graph_drain_detail::validate_projection(
          owner, invocation, batch, stage, resource, selected, materialization,
          source_region, target_region, projected)) {
    return AuthorityResult{.failure = AuthorityFailure::Invalid};
  }
  std::array<CacheKey, execution::GraphDrainCapacity> keys{};
  for (std::size_t index = 0u; index < selected.size(); ++index) {
    if (!project_graph_cache_key(materialization, selected[index].key,
                                 keys[index])) {
      return AuthorityResult{.failure = AuthorityFailure::Invalid};
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (keys[prior] == keys[index]) {
        return AuthorityResult{.failure = AuthorityFailure::Invalid};
      }
    }
  }

  std::lock_guard lock{authority_.gate_};
  const AuthorityResult destination = authority_.begin_graph_locked(
      selected, materialization, target_region, projected.epoch.ordinal);
  if (!destination) {
    return destination;
  }
  const auto rollback_destination = [&]() noexcept {
    return authority_.complete_locked(destination.lease.token, false, true);
  };
  if (destination.lease.bindings.size() != selected.size()) {
    const bool terminal = rollback_destination();
    return AuthorityResult{.failure = terminal ? AuthorityFailure::Invalid
                                               : AuthorityFailure::Busy};
  }

  const AuthorityResult source =
      authority_.begin_drain_locked(
          std::span<const CacheKey>{keys.data(), selected.size()},
          source_region.first, source_region.count, TransitionKind::Migrate);
  if (!source) {
    const bool terminal = rollback_destination();
    return AuthorityResult{.failure = terminal ? source.failure
                                               : AuthorityFailure::Busy};
  }
  const auto rollback = [&]() noexcept {
    const bool source_terminal =
        authority_.complete_locked(source.lease.token, false, false);
    const bool destination_terminal = rollback_destination();
    return source_terminal && destination_terminal;
  };
  if (source.lease.transitions.size() != selected.size()) {
    const bool terminal = rollback();
    return AuthorityResult{.failure = terminal ? AuthorityFailure::Invalid
                                               : AuthorityFailure::Busy};
  }

  std::array<execution::GraphDrainPage, execution::GraphDrainCapacity> pages{};
  for (std::size_t index = 0u; index < selected.size(); ++index) {
    const CacheTransition transition = source.lease.transitions[index];
    const CacheBinding binding = destination.lease.bindings[index];
    std::uint64_t bytes = 0u;
    if (transition.kind != TransitionKind::Migrate ||
        transition.key != keys[index] || binding.key != keys[index] ||
        transition.frame < source_region.first ||
        transition.frame - source_region.first >= source_region.count ||
        binding.frame < target_region.first ||
        binding.frame - target_region.first >= target_region.count ||
        transition.frame == binding.frame || binding.access != Access::Write ||
        !invocation.page_bytes(resource, selected[index].key.page, bytes)) {
      const bool terminal = rollback();
      return AuthorityResult{.failure = terminal ? AuthorityFailure::Invalid
                                                 : AuthorityFailure::Busy};
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (source.lease.transitions[prior].frame == transition.frame ||
          destination.lease.bindings[prior].frame == binding.frame) {
        const bool terminal = rollback();
        return AuthorityResult{.failure = terminal ? AuthorityFailure::Invalid
                                                   : AuthorityFailure::Busy};
      }
    }
    pages[index] = execution::GraphDrainPage{
        .use = selected[index].key,
        .key = keys[index],
        .bytes = bytes,
        .source_frame = transition.frame,
        .target_frame = binding.frame,
    };
  }
  if (!authority_.activate_locked(destination.lease.token)) {
    const bool terminal = rollback();
    return AuthorityResult{.failure = terminal ? AuthorityFailure::Invalid
                                               : AuthorityFailure::Busy};
  }
  ticket.owner_ = &authority_;
  ticket.plan_owner_ = std::move(owner);
  ticket.pages_ = pages;
  ticket.plan_ = ticket.plan_owner_->identity();
  ticket.completion_ = Status::fail(Reason::CompletionInvalid);
  ticket.source_token_ = source.lease.token;
  ticket.destination_token_ = destination.lease.token;
  ticket.coordinate_ = projected.epoch.ordinal;
  ticket.page_count_ = selected.size();
  ticket.terminal_ = execution::TerminalKind::Known;
  ticket.completion_may_write_ = false;
  ticket.terminalled_ = false;
  return AuthorityResult{.failure = AuthorityFailure::None};
}

} // namespace rund::compute::detail::residency
