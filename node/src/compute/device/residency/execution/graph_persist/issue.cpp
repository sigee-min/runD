#include "../../registry/graph_persist_owner.hpp"
#include "internal.hpp"

#include <algorithm>
#include <exception>
#include <limits>
#include <new>

namespace rund::compute::detail::residency {

AuthorityResult GraphPersistOwner::issue_graph_persist(
    std::shared_ptr<const ResidencyPlan> owner,
    const TiledGraphInvocation &invocation, const std::uint64_t batch,
    const std::size_t stage, const std::uint32_t resource,
    const std::span<const PageUse> selected,
    const GraphMaterialization materialization, const FrameRegion region,
    execution::GraphPersist &ticket) noexcept {
  if (ticket) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }
  if (ticket.book_domain() != 0u && !ticket.identity().valid()) {
    return AuthorityResult{.failure = AuthorityFailure::Invalid};
  }
  graph_persist_detail::Projection projected{};
  if (!graph_persist_detail::validate_projection(
          owner, invocation, batch, stage, resource, selected, materialization,
          region, projected)) {
    return AuthorityResult{.failure = AuthorityFailure::Invalid};
  }
  const std::uint64_t coordinate = projected.epoch.ordinal;
  if (coordinate == 0u) {
    return AuthorityResult{.failure = AuthorityFailure::Invalid};
  }

  std::array<CacheKey, execution::GraphPersistCapacity> keys{};
  std::array<execution::GraphPersistPage, execution::GraphPersistCapacity>
      pages{};
  for (std::size_t index = 0u; index < selected.size(); ++index) {
    const DirtyExtent dirty{.offset = selected[index].dirty.offset,
                            .bytes = selected[index].dirty.bytes};
    std::uint64_t page_bytes = 0u;
    std::uint64_t page_offset = 0u;
    std::uint64_t backing_offset = 0u;
    std::uint64_t dirty_end = 0u;
    if (!project_graph_cache_key(materialization, selected[index].key,
                                 keys[index]) ||
        !invocation.page_bytes(resource, selected[index].key.page,
                               page_bytes) ||
        !kernel::checked::add(dirty.offset, dirty.bytes, dirty_end) ||
        dirty_end > page_bytes ||
        !kernel::checked::mul(selected[index].key.page,
                              materialization.page_bytes, page_offset) ||
        !kernel::checked::add(page_offset, dirty.offset, backing_offset)) {
      return AuthorityResult{.failure = AuthorityFailure::Invalid};
    }
    pages[index] = execution::GraphPersistPage{
        .use = selected[index].key,
        .key = keys[index],
        .backing_offset = backing_offset,
        .frame_offset = dirty.offset,
        .bytes = dirty.bytes,
    };
  }

  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() ||
      authority_.execution_state_.slot.token != 0u ||
      std::any_of(authority_.cycle_state_.graph_persists.begin(),
                  authority_.cycle_state_.graph_persists.end(),
                  [](const Authority::LeaseSlot &slot) {
                    return slot.state == Authority::LeaseState::RetryReady;
                  })) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }
  const auto free =
      std::find_if(authority_.cycle_state_.graph_persists.begin(),
                   authority_.cycle_state_.graph_persists.end(),
                   [](const Authority::LeaseSlot &slot) {
                     return slot.state == Authority::LeaseState::Free;
                   });
  if (free == authority_.cycle_state_.graph_persists.end()) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }
  Authority::LeaseSlot &slot = *free;
  ::rund::compute::detail::residency::clear(slot);
  try {
    slot.undo_frames.reserve(selected.size());
    slot.undo.reserve(selected.size());
    slot.transitions.reserve(selected.size());
  } catch (const std::bad_alloc &) {
    ::rund::compute::detail::residency::clear(slot);
    return AuthorityResult{.failure = AuthorityFailure::Capacity};
  }
  if (authority_.credentials_.next_generation == 0u ||
      authority_.credentials_.next_generation ==
          std::numeric_limits<std::uint64_t>::max()) {
    ::rund::compute::detail::residency::clear(slot);
    return AuthorityResult{.failure = AuthorityFailure::Capacity};
  }
  const std::uint64_t generation = authority_.credentials_.next_generation++;
  const auto fail = [&](const AuthorityFailure failure) noexcept {
    if (!graph_persist_detail::check_core(
            authority_.frames_, slot, region,
            graph_persist_detail::UndoMode::Drain)) {
      std::terminate();
    }
    if (!slot.transitions.empty()) {
      graph_persist_detail::rollback(authority_.frames_, slot);
    }
    ::rund::compute::detail::residency::clear(slot);
    return AuthorityResult{.failure = failure};
  };
  try {
    for (std::size_t index = 0u; index < selected.size(); ++index) {
      const DirtyExtent dirty{.offset = pages[index].backing_offset,
                              .bytes = pages[index].bytes};
      std::size_t frame = authority_.frames_.size();
      for (std::size_t candidate = region.first;
           candidate < static_cast<std::size_t>(region.first) + region.count;
           ++candidate) {
        if (authority_.frames_[candidate].assigned &&
            authority_.frames_[candidate].state != FrameState::Empty &&
            authority_.frames_[candidate].key == keys[index]) {
          if (frame != authority_.frames_.size()) {
            return fail(AuthorityFailure::Invalid);
          }
          frame = candidate;
        }
      }
      if (frame == authority_.frames_.size() ||
          frame > std::numeric_limits<std::uint32_t>::max() ||
          authority_.frames_[frame].tier != FrameTier::Host ||
          authority_.frames_[frame].role != FrameRole::Output ||
          authority_.frames_[frame].state != FrameState::Dirty ||
          authority_.frames_[frame].dirty != dirty) {
        return fail(AuthorityFailure::Invalid);
      }
      const std::uint32_t physical = static_cast<std::uint32_t>(frame);
      if (std::any_of(slot.undo_frames.begin(), slot.undo_frames.end(),
                      [physical](const std::uint32_t prior) {
                        return prior == physical;
                      })) {
        return fail(AuthorityFailure::Invalid);
      }
      slot.undo_frames.push_back(physical);
      slot.undo.push_back(authority_.frames_[frame]);
      slot.transitions.push_back(CacheTransition{
          .key = keys[index],
          .frame = physical,
          .kind = TransitionKind::Writeback,
          .dirty = authority_.frames_[frame].dirty,
      });
      authority_.frames_[frame].state = FrameState::Writeback;
      pages[index].frame = physical;
    }
  } catch (const std::bad_alloc &) {
    return fail(AuthorityFailure::Capacity);
  }
  std::uint64_t token = 0u;
  if (!graph_persist_detail::mint(authority_.credentials_.next_token, token)) {
    return fail(AuthorityFailure::Capacity);
  }
  slot.token = token;
  slot.generation = generation;
  slot.coordinate = coordinate;
  slot.book_domain = ticket.book_domain();
  slot.retry_region = region;
  slot.identity = ticket.identity_;
  slot.state = Authority::LeaseState::Drain;
  ticket.owner_ = &authority_;
  ticket.plan_owner_ = std::move(owner);
  ticket.pages_ = pages;
  ticket.plan_ = ticket.plan_owner_->identity();
  slot.plan = ticket.plan_;
  ticket.completion_ = Status::fail(Reason::CompletionInvalid);
  ticket.token_ = slot.token;
  ticket.generation_ = generation;
  ticket.coordinate_ = coordinate;
  ticket.region_ = region;
  ticket.page_count_ = selected.size();
  ticket.terminal_ = execution::TerminalKind::Known;
  ticket.completion_may_write_ = false;
  ticket.terminalled_ = false;
  return AuthorityResult{.failure = AuthorityFailure::None};
}

} // namespace rund::compute::detail::residency
