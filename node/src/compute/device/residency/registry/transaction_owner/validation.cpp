#include "local.hpp"

#include <algorithm>

namespace rund::compute::detail::residency::transaction_detail {

bool virtual_key(const CacheKey &key, const std::uint64_t backing,
                 const std::uint64_t version,
                 const std::uint64_t materialization_hi,
                 const std::uint64_t materialization_lo,
                 const std::uint64_t page_count,
                 const std::uint64_t boundary_page,
                 const std::uint64_t boundary_extent) noexcept {
  const std::uint64_t extent = key.page == boundary_page ? boundary_extent : 0u;
  return key.domain == CacheDomain::Backing && key.backing == backing &&
         key.version == version && key.extent == extent &&
         key.materialization_hi == materialization_hi &&
         key.materialization_lo == materialization_lo && key.page < page_count;
}

bool valid_regions(const std::vector<Authority::Frame> &frames,
                   const std::span<const FrameRegion> regions) noexcept {
  if (regions.empty()) {
    return false;
  }
  for (std::size_t index = 0u; index < regions.size(); ++index) {
    const FrameRegion region = regions[index];
    if (region.count == 0u || region.role != FrameRole::Output ||
        region.first > frames.size() ||
        region.count > frames.size() - region.first) {
      return false;
    }
    for (std::size_t frame_index = region.first;
         frame_index < static_cast<std::size_t>(region.first) + region.count;
         ++frame_index) {
      const Authority::Frame &frame = frames[frame_index];
      if (!frame.assigned || frame.tier != region.tier ||
          frame.role != region.role) {
        return false;
      }
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      const FrameRegion previous = regions[prior];
      const std::uint64_t region_end =
          static_cast<std::uint64_t>(region.first) + region.count;
      const std::uint64_t previous_end =
          static_cast<std::uint64_t>(previous.first) + previous.count;
      const bool overlap =
          static_cast<std::uint64_t>(region.first) < previous_end &&
          static_cast<std::uint64_t>(previous.first) < region_end;
      if (overlap) {
        return false;
      }
    }
  }
  return true;
}

bool capture_rows(const std::vector<Authority::Frame> &frames,
                  const std::uint64_t backing, const std::uint64_t version,
                  const std::uint64_t materialization_hi,
                  const std::uint64_t materialization_lo,
                  const std::uint64_t page_count,
                  const std::uint64_t boundary_page,
                  const std::uint64_t boundary_extent,
                  const std::span<const FrameRegion> regions,
                  const std::span<const VirtualTransactionLease::Row> journal,
                  const std::uint64_t owner_generation, const bool require_rows,
                  VirtualTransactionLease &lease) noexcept {
  lease.row_count = 0u;
  if (regions.size() != VirtualTransactionLease::RegionCapacity ||
      journal.size() > VirtualTransactionLease::Capacity ||
      owner_generation == 0u || !valid_regions(frames, regions)) {
    return false;
  }
  const auto in_region = [&](const VirtualTransactionLease::Row &row) {
    return std::any_of(
        regions.begin(), regions.end(), [&](const FrameRegion region) {
          return region.tier == row.tier && region.role == row.role &&
                 row.frame >= region.first &&
                 static_cast<std::uint64_t>(row.frame - region.first) <
                     region.count;
        });
  };
  for (std::size_t journal_index = 0u; journal_index < journal.size();
       ++journal_index) {
    const VirtualTransactionLease::Row &wanted = journal[journal_index];
    if (wanted.frame >= frames.size() ||
        wanted.owner_generation != owner_generation ||
        wanted.role != FrameRole::Output || !in_region(wanted) ||
        !virtual_key(wanted.key, backing, version, materialization_hi,
                     materialization_lo, page_count, boundary_page,
                     boundary_extent)) {
      lease.row_count = 0u;
      return false;
    }
    for (std::size_t prior = 0u; prior < journal_index; ++prior) {
      if (journal[prior].key == wanted.key) {
        lease.row_count = 0u;
        return false;
      }
    }
    for (std::size_t prior = 0u; prior < lease.row_count; ++prior) {
      if (lease.frames[prior] == wanted.frame) {
        lease.row_count = 0u;
        return false;
      }
    }
    const Authority::Frame &frame = frames[wanted.frame];
    if (!frame.assigned || frame.tier != wanted.tier ||
        frame.role != wanted.role) {
      lease.row_count = 0u;
      return false;
    }
    if (frame.state == FrameState::Empty) {
      if (frame.transaction_generation != 0u) {
        lease.row_count = 0u;
        return false;
      }
      continue;
    }
    if (frame.key != wanted.key) {
      if (frame.transaction_generation != 0u ||
          virtual_key(frame.key, backing, version, materialization_hi,
                      materialization_lo, page_count, boundary_page,
                      boundary_extent)) {
        lease.row_count = 0u;
        return false;
      }
      continue;
    }
    if (frame.transaction_generation != owner_generation) {
      lease.row_count = 0u;
      return false;
    }
    if (frame.alias_claims != 0u ||
        (require_rows &&
         (frame.state != FrameState::Resident || !frame.dirty.empty())) ||
        lease.row_count >= VirtualTransactionLease::Capacity) {
      lease.row_count = 0u;
      return false;
    }
    lease.frames[lease.row_count++] = wanted.frame;
  }
  for (const FrameRegion region : regions) {
    for (std::size_t index = region.first;
         index < static_cast<std::size_t>(region.first) + region.count;
         ++index) {
      const Authority::Frame &frame = frames[index];
      if (frame.transaction_generation != owner_generation) {
        continue;
      }
      const bool listed =
          std::any_of(journal.begin(), journal.end(),
                      [index](const VirtualTransactionLease::Row &row) {
                        return row.frame == index;
                      });
      if (!listed || frame.alias_claims != 0u ||
          !virtual_key(frame.key, backing, version, materialization_hi,
                       materialization_lo, page_count, boundary_page,
                       boundary_extent) ||
          frame.role != FrameRole::Output || frame.tier != region.tier) {
        lease.row_count = 0u;
        return false;
      }
    }
  }
  return !require_rows || lease.row_count != 0u;
}

} // namespace rund::compute::detail::residency::transaction_detail
