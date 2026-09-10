#include "internal.hpp"

#include "../internal.hpp"

namespace rund::compute::detail::residency::lease_detail {

std::size_t select_frame(const std::vector<registry_model::Frame> &frames,
                         const std::span<const CacheUse> uses,
                         const std::size_t first, const std::size_t count,
                         const std::uint64_t epoch) noexcept {
  const std::size_t end = first + count;
  for (std::size_t index = first; index < end; ++index) {
    if (frames[index].assigned && frames[index].state == FrameState::Empty &&
        frames[index].view_commit_stamp == 0u) {
      return index;
    }
  }
  std::size_t selected = frames.size();
  for (std::size_t index = first; index < end; ++index) {
    const registry_model::Frame &frame = frames[index];
    if (!frame.assigned || frame.state == FrameState::Mapping ||
        frame.state == FrameState::Pinned ||
        frame.state == FrameState::Writeback || frame.alias_claims != 0u ||
        frame.view_commit_stamp != 0u ||
        (frame.role == FrameRole::Output && !frame.dirty.empty()) ||
        demanded(uses, frame.key) ||
        (epoch != NeverUse && frame.retain_until != NeverUse &&
         epoch <= frame.retain_until)) {
      continue;
    }
    if (selected == frames.size() ||
        frame.next_use > frames[selected].next_use ||
        (frame.next_use == frames[selected].next_use &&
         frames[selected].key < frame.key)) {
      selected = index;
    }
  }
  return selected;
}

TransitionKind retire_dirty(const CacheKey key) noexcept {
  return key.domain == CacheDomain::Backing ? TransitionKind::Writeback
                                            : TransitionKind::Discard;
}

} // namespace rund::compute::detail::residency::lease_detail
