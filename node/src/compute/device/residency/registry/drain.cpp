#include "internal.hpp"
#include "lease_state.hpp"

#include "../registry.hpp"
#include "frame.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <span>

namespace rund::compute::detail::residency {

AuthorityResult
Authority::begin_writeback(const std::span<const CacheKey> keys,
                           const std::uint32_t first_frame,
                           const std::uint32_t frame_count) noexcept {
  return begin_drain(keys, first_frame, frame_count, TransitionKind::Writeback);
}

AuthorityResult
Authority::begin_migration(const std::span<const CacheKey> keys,
                           const std::uint32_t first_frame,
                           const std::uint32_t frame_count) noexcept {
  return begin_drain(keys, first_frame, frame_count, TransitionKind::Migrate);
}

AuthorityResult
Authority::begin_discard(const std::span<const CacheKey> keys,
                         const std::uint32_t first_frame,
                         const std::uint32_t frame_count) noexcept {
  return begin_drain(keys, first_frame, frame_count, TransitionKind::Discard);
}

AuthorityResult Authority::begin_drain(const std::span<const CacheKey> keys,
                                       const std::uint32_t first_frame,
                                       const std::uint32_t frame_count,
                                       const TransitionKind kind) noexcept {
  std::lock_guard lock{gate_};
  return begin_drain_locked(keys, first_frame, frame_count, kind);
}

AuthorityResult Authority::begin_drain_locked(
    const std::span<const CacheKey> keys, const std::uint32_t first_frame,
    const std::uint32_t frame_count, const TransitionKind kind) noexcept {
  if (view_commit_quarantined_locked() || execution_state_.slot.token != 0u ||
      retry_ready(cycle_state_.graph_persists)) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }
  const std::size_t first = first_frame;
  const std::size_t count = frame_count;
  if ((kind != TransitionKind::Writeback && kind != TransitionKind::Migrate &&
       kind != TransitionKind::Discard) ||
      active(cycle_state_.writeback) || frames_.empty() || keys.empty() ||
      count == 0u || first > frames_.size() || count > frames_.size() - first) {
    return AuthorityResult{.failure = active(cycle_state_.writeback)
                                          ? AuthorityFailure::Busy
                                          : AuthorityFailure::Invalid};
  }
  clear(cycle_state_.writeback);
  try {
    for (const CacheKey key : keys) {
      const std::size_t frame = find_frame(frames_, key, first, count);
      if (frame == frames_.size() ||
          frames_[frame].state != FrameState::Dirty ||
          frames_[frame].dirty.empty() ||
          (kind == TransitionKind::Migrate &&
           (frames_[frame].tier != FrameTier::Device ||
            frames_[frame].role != FrameRole::Output)) ||
          (kind == TransitionKind::Writeback &&
           frames_[frame].key.domain != CacheDomain::Backing)) {
        rollback(frames_, cycle_state_.writeback, false);
        clear(cycle_state_.writeback);
        return AuthorityResult{.failure = AuthorityFailure::Invalid};
      }
      save(cycle_state_.writeback, static_cast<std::uint32_t>(frame),
           frames_[frame]);
      cycle_state_.writeback.transitions.push_back(CacheTransition{
          .key = key,
          .frame = static_cast<std::uint32_t>(frame),
          .kind = kind,
          .dirty = frames_[frame].dirty,
      });
      frames_[frame].state = FrameState::Writeback;
    }
  } catch (const std::bad_alloc &) {
    rollback(frames_, cycle_state_.writeback, false);
    clear(cycle_state_.writeback);
    return AuthorityResult{.failure = AuthorityFailure::Capacity};
  }
  const std::uint64_t token = next_token(credentials_.next_token);
  if (token == 0u) {
    rollback(frames_, cycle_state_.writeback, false);
    clear(cycle_state_.writeback);
    return AuthorityResult{.failure = AuthorityFailure::Capacity};
  }
  cycle_state_.writeback.token = token;
  cycle_state_.writeback.state = LeaseState::Drain;
  return AuthorityResult{.failure = AuthorityFailure::None,
                         .lease = EpochLease{
                             .bindings = cycle_state_.writeback.bindings,
                             .transitions = cycle_state_.writeback.transitions,
                             .token = cycle_state_.writeback.token,
                         }};
}

AuthorityResult
Authority::drain_dirty(const std::uint32_t first_frame,
                       const std::uint32_t frame_count) noexcept {
  std::lock_guard lock{gate_};
  if (view_commit_quarantined_locked() || execution_state_.slot.token != 0u) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }
  const bool epoch_active = std::any_of(cycle_state_.epochs.begin(),
                                        cycle_state_.epochs.end(), active);
  const std::size_t first = first_frame;
  const std::size_t count = frame_count;
  if (active(cycle_state_.writeback) ||
      any_active(cycle_state_.graph_persists) || epoch_active ||
      frames_.empty() || count == 0u || first > frames_.size() ||
      count > frames_.size() - first) {
    return AuthorityResult{
        .failure = active(cycle_state_.writeback) ||
                           any_active(cycle_state_.graph_persists) ||
                           epoch_active
                       ? AuthorityFailure::Busy
                       : AuthorityFailure::Invalid};
  }
  clear(cycle_state_.writeback);
  try {
    for (std::size_t index = first; index < first + count; ++index) {
      if (frames_[index].dirty.empty()) {
        continue;
      }
      if (frames_[index].key.domain != CacheDomain::Backing) {
        rollback(frames_, cycle_state_.writeback, false);
        clear(cycle_state_.writeback);
        return AuthorityResult{.failure = AuthorityFailure::Invalid};
      }
      save(cycle_state_.writeback, static_cast<std::uint32_t>(index),
           frames_[index]);
      cycle_state_.writeback.transitions.push_back(CacheTransition{
          .key = frames_[index].key,
          .frame = static_cast<std::uint32_t>(index),
          .kind = TransitionKind::Writeback,
          .dirty = frames_[index].dirty,
      });
      frames_[index].state = FrameState::Writeback;
    }
  } catch (const std::bad_alloc &) {
    rollback(frames_, cycle_state_.writeback, false);
    clear(cycle_state_.writeback);
    return AuthorityResult{.failure = AuthorityFailure::Capacity};
  }
  const std::uint64_t token = next_token(credentials_.next_token);
  if (token == 0u) {
    rollback(frames_, cycle_state_.writeback, false);
    clear(cycle_state_.writeback);
    return AuthorityResult{.failure = AuthorityFailure::Capacity};
  }
  cycle_state_.writeback.token = token;
  cycle_state_.writeback.state = LeaseState::Drain;
  cycle_state_.writeback.terminal = true;
  return AuthorityResult{.failure = AuthorityFailure::None,
                         .lease = EpochLease{
                             .bindings = cycle_state_.writeback.bindings,
                             .transitions = cycle_state_.writeback.transitions,
                             .token = cycle_state_.writeback.token,
                         }};
}

bool Authority::complete_migration(
    const std::uint64_t source_token,
    const std::uint64_t destination_token) noexcept {
  std::lock_guard lock{gate_};
  if (view_commit_quarantined_locked() || source_token == 0u ||
      retry_ready(cycle_state_.graph_persists) || destination_token == 0u ||
      cycle_state_.writeback.token != source_token ||
      cycle_state_.writeback.state != LeaseState::Drain) {
    return false;
  }
  const auto destination =
      std::find_if(cycle_state_.epochs.begin(), cycle_state_.epochs.end(),
                   [destination_token](const LeaseSlot &slot) {
                     return slot.token == destination_token;
                   });
  if (destination == cycle_state_.epochs.end() ||
      destination->state != LeaseState::Computing ||
      destination->bindings.empty() ||
      cycle_state_.writeback.transitions.size() !=
          destination->bindings.size() ||
      destination->undo_frames.size() != destination->bindings.size() ||
      destination->undo.size() != destination->bindings.size() ||
      std::any_of(destination->transitions.begin(),
                  destination->transitions.end(),
                  [](const CacheTransition transition) {
                    return transition.kind != TransitionKind::Map &&
                           transition.kind != TransitionKind::Unmap;
                  })) {
    return false;
  }
  for (std::size_t index = 0u;
       index < cycle_state_.writeback.transitions.size(); ++index) {
    const CacheTransition transition =
        cycle_state_.writeback.transitions[index];
    const CacheBinding binding = destination->bindings[index];
    const Frame &prior = destination->undo[index];
    if (transition.kind != TransitionKind::Migrate ||
        transition.frame >= frames_.size() || binding.frame >= frames_.size() ||
        transition.frame == binding.frame ||
        destination->undo_frames[index] != binding.frame ||
        !prior.dirty.empty() ||
        (prior.state != FrameState::Empty &&
         prior.state != FrameState::Resident)) {
      return false;
    }
    const Frame &source = frames_[transition.frame];
    const Frame &target = frames_[binding.frame];
    // A migration is not a generic two-token commit. It must be the exact
    // ordered Device-output epoch copied into the corresponding Host-output
    // epoch, including the backing identity and dirty byte interval.
    if (!source.assigned || source.tier != FrameTier::Device ||
        source.role != FrameRole::Output || source.key != transition.key ||
        source.state != FrameState::Writeback || source.dirty.empty() ||
        source.dirty != transition.dirty || source.dirty != binding.dirty ||
        !target.assigned || target.tier != FrameTier::Host ||
        target.role != FrameRole::Output || target.key != binding.key ||
        (source.transaction_generation == 0u) !=
            (target.transaction_generation == 0u) ||
        (source.transaction_generation != 0u &&
         source.transaction_generation != target.transaction_generation) ||
        (target.state != FrameState::Pinned &&
         target.state != FrameState::Mapping) ||
        !writes(binding.access) || source.key != target.key) {
      return false;
    }
  }
  for (const CacheTransition &transition : cycle_state_.writeback.transitions) {
    frames_[transition.frame] = frame_detail::empty(frames_[transition.frame]);
  }
  for (const CacheBinding &binding : destination->bindings) {
    Frame &frame = frames_[binding.frame];
    frame.dirty = binding.dirty;
    frame.state = FrameState::Dirty;
  }
  clear(cycle_state_.writeback);
  clear(*destination);
  return true;
}

bool Authority::discard(const std::uint64_t token) noexcept {
  std::lock_guard lock{gate_};
  if (view_commit_quarantined_locked() || token == 0u ||
      retry_ready(cycle_state_.graph_persists) ||
      cycle_state_.writeback.token != token ||
      cycle_state_.writeback.state != LeaseState::Drain) {
    return false;
  }
  rollback(frames_, cycle_state_.writeback, false);
  for (const CacheTransition &transition : cycle_state_.writeback.transitions) {
    if (transition.frame < frames_.size() &&
        (transition.kind == TransitionKind::Writeback ||
         transition.kind == TransitionKind::Migrate ||
         transition.kind == TransitionKind::Discard) &&
        frames_[transition.frame].key == transition.key) {
      frames_[transition.frame] =
          frame_detail::empty(frames_[transition.frame]);
    }
  }
  clear(cycle_state_.writeback);
  return true;
}

} // namespace rund::compute::detail::residency
