#include "internal.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::compute::detail::residency {

namespace view_plan_detail {

bool validate_graph(
    const std::vector<Authority::Frame> &frames,
    const TiledGraphInvocation &schedule,
    const std::span<const GraphFreezeRequest> requests) noexcept {
  if (frames.empty() || requests.empty() ||
      requests.size() > registry_model::ViewCommitPlan::GraphRequestCapacity ||
      schedule.page_count() == 0u) {
    return false;
  }
  for (std::size_t request_index = 0u; request_index < requests.size();
       ++request_index) {
    const GraphFreezeRequest &request = requests[request_index];
    const GraphMaterialization &materialization = request.materialization;
    if (request.regions.empty() || materialization.resource == 0u ||
        request.regions.size() > GraphFreezeRequest::RegionCapacity ||
        materialization.key.domain != CacheDomain::Backing ||
        materialization.key.backing == 0u || materialization.page_bytes == 0u ||
        materialization.page_count != schedule.page_count() ||
        materialization.key.page != 0u) {
      return false;
    }
    for (std::size_t region_index = 0u; region_index < request.regions.size();
         ++region_index) {
      const FrameRegion region = request.regions[region_index];
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
      for (std::size_t prior_request = 0u; prior_request <= request_index;
           ++prior_request) {
        const std::span<const FrameRegion> prior_regions =
            requests[prior_request].regions;
        const std::size_t prior_count = prior_request == request_index
                                            ? region_index
                                            : prior_regions.size();
        for (std::size_t prior_region = 0u; prior_region < prior_count;
             ++prior_region) {
          const FrameRegion other = prior_regions[prior_region];
          const std::uint64_t region_end =
              static_cast<std::uint64_t>(region.first) + region.count;
          const std::uint64_t other_end =
              static_cast<std::uint64_t>(other.first) + other.count;
          if (region.first < other_end && other.first < region_end) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

void apply_graph(std::vector<Authority::Frame> &frames,
                 const TiledGraphInvocation &schedule,
                 const std::span<const GraphFreezeRequest> requests) noexcept {
  for (const GraphFreezeRequest &request : requests) {
    for (const FrameRegion region : request.regions) {
      for (std::size_t index = region.first;
           index < static_cast<std::size_t>(region.first) + region.count;
           ++index) {
        Authority::Frame &frame = frames[index];
        const CacheKey &key = frame.key;
        const GraphMaterialization &materialization = request.materialization;
        const bool same_dataset =
            key.domain == materialization.key.domain &&
            key.backing == materialization.key.backing &&
            key.version == materialization.key.version &&
            key.materialization_hi == materialization.key.materialization_hi &&
            key.materialization_lo == materialization.key.materialization_lo;
        const std::uint64_t expected_extent =
            key.page == materialization.page_count - 1u
                ? materialization.boundary_extent
                : 0u;
        std::uint64_t first_use = NeverUse;
        const bool planned =
            schedule.first_use(materialization.resource, key.page, first_use);
        frame.next_use = frame.state != FrameState::Empty && same_dataset &&
                                 planned && key.extent == expected_extent
                             ? first_use
                             : NeverUse;
        frame.retain_until = NeverUse;
      }
    }
  }
}

} // namespace view_plan_detail

} // namespace rund::compute::detail::residency
