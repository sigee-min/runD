#include "../../registry.hpp"
#include "../internal.hpp"
#include "../lease_state.hpp"
#include "internal.hpp"

#include <algorithm>
#include <limits>
#include <new>

namespace rund::compute::detail::residency {

AuthorityResult Authority::begin(const std::span<const CacheUse> uses,
                                 const FrameTier tier, const FrameRole role,
                                 const std::uint32_t first_frame,
                                 const std::uint32_t frame_count,
                                 const CpuReservationKey cpu_key) noexcept {
  std::lock_guard lock{gate_};
  return begin_locked(uses, tier, role, first_frame, frame_count, cpu_key);
}

AuthorityResult Authority::begin_locked(
    const std::span<const CacheUse> uses, const FrameTier tier,
    const FrameRole role, const std::uint32_t first_frame,
    const std::uint32_t frame_count, const CpuReservationKey cpu_key) noexcept {
  if (view_commit_quarantined_locked() ||
      retry_ready(cycle_state_.graph_persists)) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }
  if (supplied_cpu_key(cpu_key) &&
      (!valid_cpu_key(cpu_key) || cpu_key.owner() != credentials_.owner_id ||
       cpu_graph_state_.pending_cpu != cpu_key)) {
    return AuthorityResult{.failure = AuthorityFailure::Invalid};
  }
  if (!supplied_cpu_key(cpu_key) && cpu_graph_state_.pending_cpu) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }
  registry_model::PendingCpu pending{this, cpu_key, supplied_cpu_key(cpu_key)};
  if (execution_state_.slot.token != 0u) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }
  if (std::any_of(cycle_state_.epochs.begin(), cycle_state_.epochs.end(),
                  [](const LeaseSlot &slot) {
                    return slot.state == LeaseState::Prepared;
                  })) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }
  if (supplied_cpu_key(cpu_key) &&
      std::any_of(cycle_state_.epochs.begin(), cycle_state_.epochs.end(),
                  [cpu_key](const LeaseSlot &slot) {
                    return slot.state != LeaseState::Free &&
                           slot.cpu_key == cpu_key;
                  })) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }
  const std::size_t first = first_frame;
  const std::size_t count = frame_count;
  if (frames_.empty() || uses.empty() || count == 0u ||
      first > frames_.size() || count > frames_.size() - first ||
      uses.size() > count) {
    return AuthorityResult{.failure = uses.size() > count
                                          ? AuthorityFailure::Capacity
                                          : AuthorityFailure::Invalid};
  }
  if (std::any_of(frames_.begin() + static_cast<std::ptrdiff_t>(first),
                  frames_.begin() + static_cast<std::ptrdiff_t>(first + count),
                  [tier, role](const Frame &frame) {
                    return !frame.assigned || frame.tier != tier ||
                           frame.role != role;
                  })) {
    return AuthorityResult{.failure = AuthorityFailure::Invalid};
  }
  for (std::size_t left = 0u; left < uses.size(); ++left) {
    const CacheUse &use = uses[left];
    if (use.key.backing == 0u || writes(use.access) == use.dirty.empty() ||
        ((use.epoch == NeverUse) != (use.retain_until == NeverUse)) ||
        (use.epoch != NeverUse && use.retain_until < use.epoch) ||
        (left != 0u && use.epoch != uses.front().epoch) ||
        (!use.dirty.empty() &&
         use.dirty.bytes >
             std::numeric_limits<std::uint64_t>::max() - use.dirty.offset)) {
      return AuthorityResult{.failure = AuthorityFailure::Invalid};
    }
    for (std::size_t right = left + 1u; right < uses.size(); ++right) {
      if (uses[left].key == uses[right].key) {
        return AuthorityResult{.failure = AuthorityFailure::Invalid};
      }
    }
  }

  const auto free = std::find_if(
      cycle_state_.epochs.begin(), cycle_state_.epochs.end(),
      [](const LeaseSlot &slot) { return slot.state == LeaseState::Free; });
  if (free == cycle_state_.epochs.end()) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }
  LeaseSlot &slot = *free;
  clear(slot);
  try {
    for (const CacheUse &use : uses) {
      std::size_t frame = find_frame(frames_, use.key, first, count);
      if (frame != frames_.size() &&
          (frames_[frame].state == FrameState::Mapping ||
           frames_[frame].state == FrameState::Pinned ||
           frames_[frame].state == FrameState::Writeback)) {
        rollback(frames_, slot, false);
        clear(slot);
        return AuthorityResult{.failure = AuthorityFailure::Busy};
      }
      bool fetch = false;
      DirtyExtent prior_dirty{};
      if (frame == frames_.size()) {
        frame =
            lease_detail::select_frame(frames_, uses, first, count, use.epoch);
        if (frame == frames_.size()) {
          rollback(frames_, slot, false);
          clear(slot);
          return AuthorityResult{.failure = AuthorityFailure::Capacity};
        }
        save(slot, static_cast<std::uint32_t>(frame), frames_[frame]);
        prior_dirty = frames_[frame].dirty;
        if (frames_[frame].state != FrameState::Empty) {
          if (!frames_[frame].dirty.empty()) {
            slot.transitions.push_back(CacheTransition{
                .key = frames_[frame].key,
                .frame = static_cast<std::uint32_t>(frame),
                .kind = lease_detail::retire_dirty(frames_[frame].key),
                .dirty = frames_[frame].dirty,
            });
          }
          slot.transitions.push_back(CacheTransition{
              .key = frames_[frame].key,
              .frame = static_cast<std::uint32_t>(frame),
              .kind = TransitionKind::Unmap,
          });
        }
        if (reads(use.access) && use.key.domain == CacheDomain::Transient) {
          rollback(frames_, slot, false);
          clear(slot);
          return AuthorityResult{.failure = AuthorityFailure::Invalid};
        }
        if (reads(use.access)) {
          slot.transitions.push_back(CacheTransition{
              .key = use.key,
              .frame = static_cast<std::uint32_t>(frame),
              .kind = TransitionKind::Fetch,
          });
        }
        slot.transitions.push_back(CacheTransition{
            .key = use.key,
            .frame = static_cast<std::uint32_t>(frame),
            .kind = TransitionKind::Map,
        });
        const FrameTier frame_tier = frames_[frame].tier;
        const FrameRole frame_role = frames_[frame].role;
        const std::uint64_t extent = frames_[frame].extent;
        const std::uint64_t view = frames_[frame].view;
        frames_[frame] = Frame{
            .key = use.key,
            .next_use = use.next_use,
            .retain_until = use.retain_until,
            .state = FrameState::Mapping,
            .tier = frame_tier,
            .role = frame_role,
            .extent = extent,
            .view = view,
            .assigned = true,
        };
        fetch = reads(use.access);
      } else {
        save(slot, static_cast<std::uint32_t>(frame), frames_[frame]);
        prior_dirty = frames_[frame].dirty;
        frames_[frame].next_use = use.next_use;
        frames_[frame].retain_until = use.retain_until;
        frames_[frame].state = FrameState::Pinned;
      }
      slot.bindings.push_back(CacheBinding{
          .key = use.key,
          .frame = static_cast<std::uint32_t>(frame),
          .access = use.access,
          .dirty = use.dirty,
          .prior_dirty = prior_dirty,
          .fetch = fetch,
      });
    }
  } catch (const std::bad_alloc &) {
    rollback(frames_, slot, false);
    clear(slot);
    return AuthorityResult{.failure = AuthorityFailure::Capacity};
  }
  slot.token = next_token(credentials_.next_token);
  if (slot.token == 0u) {
    rollback(frames_, slot, false);
    clear(slot);
    return AuthorityResult{.failure = AuthorityFailure::Capacity};
  }
  slot.generation = next_generation(credentials_.next_generation);
  if (slot.generation == 0u) {
    rollback(frames_, slot, false);
    clear(slot);
    return AuthorityResult{.failure = AuthorityFailure::Capacity};
  }
  slot.cpu_key = cpu_key;
  slot.cpu_bound = false;
  slot.state = LeaseState::Prepared;
  pending.publish();
  return AuthorityResult{.failure = AuthorityFailure::None,
                         .lease = EpochLease{
                             .bindings = slot.bindings,
                             .transitions = slot.transitions,
                             .token = slot.token,
                             .generation = slot.generation,
                         }};
}

} // namespace rund::compute::detail::residency
