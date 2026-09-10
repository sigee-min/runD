#include "../../registry.hpp"
#include "../frame.hpp"
#include "../internal.hpp"
#include "../lease_state.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <new>
#include <span>

namespace rund::compute::detail::residency {
bool Authority::activate(const std::uint64_t token) noexcept {
  std::lock_guard lock{gate_};
  return activate_locked(token);
}

bool Authority::activate_locked(const std::uint64_t token) noexcept {
  if (view_commit_quarantined_locked() ||
      retry_ready(cycle_state_.graph_persists)) {
    return false;
  }
  const auto found = std::find_if(
      cycle_state_.epochs.begin(), cycle_state_.epochs.end(),
      [token](const LeaseSlot &slot) { return slot.token == token; });
  if (token == 0u || found == cycle_state_.epochs.end() ||
      found->state != LeaseState::Prepared ||
      (found->cpu_key && !found->cpu_bound)) {
    return false;
  }
  for (const std::uint32_t frame : found->relocation_frames) {
    if (frame >= frames_.size() || !frames_[frame].assigned ||
        (frames_[frame].state != FrameState::Pinned &&
         frames_[frame].state != FrameState::Mapping)) {
      return false;
    }
  }
  for (const CacheBinding &binding : found->bindings) {
    if (binding.frame >= frames_.size() ||
        (frames_[binding.frame].state != FrameState::Mapping &&
         frames_[binding.frame].state != FrameState::Pinned) ||
        (!binding.relocated && frames_[binding.frame].key != binding.key)) {
      return false;
    }
  }
  for (const std::uint32_t frame_index : found->relocation_frames) {
    const bool target =
        std::any_of(found->bindings.begin(), found->bindings.end(),
                    [frame_index](const CacheBinding &binding) {
                      return binding.frame == frame_index;
                    });
    if (!target) {
      Frame &frame = frames_[frame_index];
      frame = frame_detail::empty(frame);
    }
  }
  for (const CacheBinding &binding : found->bindings) {
    Frame &frame = frames_[binding.frame];
    if (binding.relocated) {
      const FrameTier tier = frame.tier;
      const FrameRole role = frame.role;
      const bool assigned = frame.assigned;
      const std::uint64_t extent = frame.extent;
      const std::uint64_t view = frame.view;
      frame = Frame{.key = binding.key,
                    .next_use = binding.next_use,
                    .retain_until = binding.retain_until,
                    .state = FrameState::Pinned,
                    .tier = tier,
                    .role = role,
                    .dirty = binding.prior_dirty,
                    .extent = extent,
                    .view = view,
                    .assigned = assigned};
    } else {
      frame.state = FrameState::Pinned;
    }
  }
  found->state = LeaseState::Computing;
  return true;
}

AuthorityResult Authority::resume(const std::uint64_t token) noexcept {
  std::lock_guard lock{gate_};
  return resume_locked(token);
}

AuthorityResult Authority::resume_locked(const std::uint64_t token) noexcept {
  if (view_commit_quarantined_locked() ||
      retry_ready(cycle_state_.graph_persists)) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }
  const auto found = std::find_if(
      cycle_state_.epochs.begin(), cycle_state_.epochs.end(),
      [token](const LeaseSlot &slot) { return slot.token == token; });
  if (token == 0u || found == cycle_state_.epochs.end() ||
      found->state != LeaseState::Computing || found->cycle != 0u ||
      (found->cpu_key && !found->cpu_bound)) {
    return AuthorityResult{.failure = AuthorityFailure::Invalid};
  }
  return AuthorityResult{
      .failure = AuthorityFailure::None,
      .lease = EpochLease{.bindings = found->bindings,
                          .transitions = found->transitions,
                          .ports = found->ports,
                          .relocations = found->relocations,
                          .remaps = found->remaps,
                          .token = found->token,
                          .generation = found->generation},
  };
}

} // namespace rund::compute::detail::residency
