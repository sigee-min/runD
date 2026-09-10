#include "internal.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::compute::detail::residency {

namespace view_plan_detail {

bool validate_stream(const std::vector<Authority::Frame> &frames,
                     const StreamPlan &schedule, const CacheKey last,
                     const std::span<const FrameRegion> regions) noexcept {
  if (frames.empty() || regions.empty() || schedule.page_count() == 0u ||
      last.backing == 0u || last.page != schedule.page_count() - 1u) {
    return false;
  }
  for (std::size_t index = 0u; index < regions.size(); ++index) {
    const FrameRegion region = regions[index];
    if (region.role != FrameRole::Input || region.count == 0u ||
        region.first > frames.size() ||
        region.count > frames.size() - region.first) {
      return false;
    }
    for (std::size_t frame = region.first;
         frame < static_cast<std::size_t>(region.first) + region.count;
         ++frame) {
      if (!frames[frame].assigned || frames[frame].tier != region.tier ||
          frames[frame].role != region.role ||
          frames[frame].view_commit_stamp != 0u ||
          frames[frame].direct_registration != nullptr) {
        return false;
      }
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      const FrameRegion other = regions[prior];
      const std::uint64_t region_end =
          static_cast<std::uint64_t>(region.first) + region.count;
      const std::uint64_t other_end =
          static_cast<std::uint64_t>(other.first) + other.count;
      if (region.first < other_end && other.first < region_end) {
        return false;
      }
    }
  }
  return true;
}

void apply_stream(std::vector<Authority::Frame> &frames,
                  const StreamPlan &schedule, const CacheKey last,
                  const std::span<const FrameRegion> regions) noexcept {
  for (const FrameRegion region : regions) {
    for (std::size_t index = region.first;
         index < static_cast<std::size_t>(region.first) + region.count;
         ++index) {
      Authority::Frame &frame = frames[index];
      const CacheKey &key = frame.key;
      const bool same_dataset =
          key.backing == last.backing && key.version == last.version &&
          key.materialization_hi == last.materialization_hi &&
          key.materialization_lo == last.materialization_lo;
      const std::uint64_t expected_extent =
          key.page == schedule.page_count() - 1u ? last.extent : 0u;
      std::uint64_t first_use = NeverUse;
      const bool planned = schedule.first_use(key.page, first_use);
      frame.next_use = frame.state != FrameState::Empty && same_dataset &&
                               planned && key.extent == expected_extent
                           ? first_use
                           : NeverUse;
      frame.retain_until = NeverUse;
    }
  }
}

} // namespace view_plan_detail

} // namespace rund::compute::detail::residency
