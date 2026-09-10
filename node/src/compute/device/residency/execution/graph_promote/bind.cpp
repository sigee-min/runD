#include "internal.hpp"

#include <algorithm>

namespace rund::compute::detail::residency {
namespace {

[[nodiscard]] bool
contains_use(const graph_promote_detail::Projection &projection,
             const execution::GraphForecastPage host) noexcept {
  const auto begin = projection.uses.begin() +
                     static_cast<std::ptrdiff_t>(projection.first_use);
  const auto end = begin + static_cast<std::ptrdiff_t>(projection.page_count);
  return std::find_if(begin, end, [host](const PageUse use) {
           return use.key == host.use;
         }) != end;
}

} // namespace

AuthorityResult GraphPromoteOwner::bind_graph_promote_group(
    const TiledGraphInvocation &invocation,
    const std::uint64_t destination_token,
    const graph_promote_detail::Group &group,
    execution::GraphPromote &ticket) noexcept {
  if (ticket || group.owner == nullptr || group.source_count == 0u ||
      group.source_count > execution::GraphPromoteSourceCapacity) {
    return AuthorityResult{.failure = AuthorityFailure::Invalid};
  }
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() ||
      std::any_of(authority_.cycle_state_.graph_persists.begin(),
                  authority_.cycle_state_.graph_persists.end(),
                  [](const registry_model::LeaseSlot &slot) {
                    return slot.state == registry_model::LeaseState::RetryReady;
                  })) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }
  const auto destination =
      std::find_if(authority_.cycle_state_.epochs.begin(),
                   authority_.cycle_state_.epochs.end(),
                   [destination_token](const registry_model::LeaseSlot &slot) {
                     return slot.token == destination_token;
                   });
  if (destination == authority_.cycle_state_.epochs.end() ||
      destination->state != registry_model::LeaseState::Prepared ||
      destination->cycle != 0u) {
    return AuthorityResult{.failure = AuthorityFailure::Invalid};
  }

  std::array<execution::GraphPromotePage, execution::GraphPromoteCapacity>
      pages{};
  std::array<std::uint64_t, execution::GraphPromoteSourceCapacity>
      source_tokens{};
  std::size_t page_count = 0u;
  for (std::size_t source_index = 0u; source_index < group.source_count;
       ++source_index) {
    const graph_promote_detail::Projection &projection =
        group.projections[source_index];
    const std::uint32_t resource = group.resources[source_index];
    const std::uint64_t source_token = group.tokens[source_index];
    const auto source =
        source_token == 0u
            ? authority_.cycle_state_.epochs.end()
            : std::find_if(
                  authority_.cycle_state_.epochs.begin(),
                  authority_.cycle_state_.epochs.end(),
                  [source_token](const registry_model::LeaseSlot &slot) {
                    return slot.token == source_token;
                  });
    if ((source_token != 0u &&
         (source == authority_.cycle_state_.epochs.end() ||
          source->state != registry_model::LeaseState::Computing ||
          source->cycle != 0u)) ||
        (source_token != 0u &&
         std::find(source_tokens.begin(), source_tokens.begin() + source_index,
                   source_token) != source_tokens.begin() + source_index)) {
      return AuthorityResult{.failure = AuthorityFailure::Invalid};
    }
    const auto destination_port = std::find_if(
        destination->ports.begin(), destination->ports.end(),
        [resource](const GraphLeasePort &port) {
          return port.resource == resource && port.access == Access::Read;
        });
    if (destination_port == destination->ports.end() ||
        destination_port->binding_count != projection.page_count ||
        destination_port->first_binding > destination->bindings.size() ||
        destination_port->binding_count >
            destination->bindings.size() - destination_port->first_binding) {
      return AuthorityResult{.failure = AuthorityFailure::Invalid};
    }

    const auto forecast_pages = group.pages[source_index];
    for (std::size_t index = 0u; index < projection.page_count; ++index) {
      const PageUse use = projection.uses[projection.first_use + index];
      const CacheBinding binding =
          destination->bindings[destination_port->first_binding + index];
      const auto host =
          std::find_if(forecast_pages.begin(), forecast_pages.end(),
                       [use](const execution::GraphForecastPage &page) {
                         return page.use == use.key;
                       });
      if (binding.key.page != use.key.page || binding.access != Access::Read ||
          binding.frame >= authority_.frames_.size()) {
        return AuthorityResult{.failure = AuthorityFailure::Invalid};
      }
      const registry_model::Frame &target = authority_.frames_[binding.frame];
      if (!target.assigned || target.tier != FrameTier::Device ||
          target.role != FrameRole::Input ||
          (target.state != FrameState::Mapping &&
           target.state != FrameState::Pinned) ||
          (!binding.relocated && target.key != binding.key)) {
        return AuthorityResult{.failure = AuthorityFailure::Invalid};
      }
      if (!binding.fetch) {
        if (host != forecast_pages.end() && host->key != binding.key) {
          return AuthorityResult{.failure = AuthorityFailure::Invalid};
        }
        continue;
      }
      if (host == forecast_pages.end() || host->key != binding.key ||
          host->frame >= authority_.frames_.size()) {
        return AuthorityResult{.failure = AuthorityFailure::Invalid};
      }
      const registry_model::Frame &source_frame =
          authority_.frames_[host->frame];
      std::uint64_t bytes = 0u;
      bool source_live = source_frame.state == FrameState::Resident &&
                         source_frame.retain_until != NeverUse &&
                         source_frame.retain_until >= projection.epoch.ordinal;
      if (source != authority_.cycle_state_.epochs.end()) {
        source_live =
            source_frame.state == FrameState::Pinned &&
            std::find_if(source->bindings.begin(), source->bindings.end(),
                         [host](const CacheBinding &value) {
                           return value.key == host->key &&
                                  value.frame == host->frame &&
                                  value.access == Access::Read;
                         }) != source->bindings.end();
      }
      if (!source_live || !source_frame.assigned ||
          source_frame.tier != FrameTier::Host ||
          source_frame.role != FrameRole::Input ||
          source_frame.key != host->key || host->frame == binding.frame ||
          !invocation.page_bytes(resource, use.key.page, bytes) ||
          bytes == 0u) {
        return AuthorityResult{.failure = AuthorityFailure::Invalid};
      }
      const TiledGraphResource *const declared =
          group.owner->tiled_graph().resource(resource);
      if (declared == nullptr || declared->page_bytes == 0u ||
          page_count >= pages.size()) {
        return AuthorityResult{.failure = AuthorityFailure::Capacity};
      }
      pages[page_count++] = execution::GraphPromotePage{
          .use = use.key,
          .key = binding.key,
          .bytes = declared->page_bytes,
          .source_frame = host->frame,
          .target_frame = binding.frame,
      };
    }
    if (std::any_of(forecast_pages.begin(), forecast_pages.end(),
                    [&projection](const auto page) {
                      return !contains_use(projection, page);
                    })) {
      return AuthorityResult{.failure = AuthorityFailure::Invalid};
    }
    source_tokens[source_index] = source_token;
  }

  ticket.owner_ = &authority_;
  ticket.plan_owner_ = group.owner;
  ticket.pages_ = pages;
  ticket.source_tokens_ = source_tokens;
  ticket.plan_ = group.owner->identity();
  ticket.completion_ = Status::fail(Reason::CompletionInvalid);
  ticket.destination_token_ = destination_token;
  ticket.coordinate_ = group.projections.front().epoch.ordinal;
  ticket.page_count_ = page_count;
  ticket.source_count_ = group.source_count;
  ticket.terminal_ = execution::TerminalKind::Known;
  ticket.completion_may_write_ = false;
  ticket.terminalled_ = false;
  return AuthorityResult{.failure = AuthorityFailure::None};
}

} // namespace rund::compute::detail::residency
