#include "../../registry.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace rund::compute::detail::residency {

[[nodiscard]] std::size_t
find_frame(const std::vector<Authority::Frame> &frames, const CacheKey key,
           const std::size_t first, const std::size_t count) noexcept {
  const std::size_t end = first + count;
  for (std::size_t index = first; index < end; ++index) {
    if (frames[index].assigned && frames[index].state != FrameState::Empty &&
        frames[index].view_commit_stamp == 0u && frames[index].key == key) {
      return index;
    }
  }
  return frames.size();
}

[[nodiscard]] bool in_region(const FrameRegion region,
                             const std::size_t frame) noexcept {
  return region.count != 0u && frame >= region.first &&
         static_cast<std::uint64_t>(frame - region.first) < region.count;
}

[[nodiscard]] bool in_regions(const std::span<const FrameRegion> regions,
                              const std::size_t frame) noexcept {
  return std::any_of(
      regions.begin(), regions.end(),
      [frame](const FrameRegion region) { return in_region(region, frame); });
}

[[nodiscard]] bool demanded(const std::span<const CacheUse> uses,
                            const CacheKey key) noexcept {
  return std::any_of(uses.begin(), uses.end(),
                     [key](const CacheUse use) { return use.key == key; });
}

[[nodiscard]] bool valid_cpu_key(const CpuReservationKey key) noexcept {
  return static_cast<bool>(key);
}

[[nodiscard]] bool supplied_cpu_key(const CpuReservationKey key) noexcept {
  return !(key == CpuReservationKey{});
}

[[nodiscard]] std::uint64_t next_token(std::uint64_t &next) noexcept {
  constexpr std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  if (next == 0u || next == max) {
    return 0u;
  }
  return next++;
}

[[nodiscard]] std::uint64_t next_generation(std::uint64_t &next) noexcept {
  if (next == 0u || next == std::numeric_limits<std::uint64_t>::max()) {
    return 0u;
  }
  return next++;
}

} // namespace rund::compute::detail::residency
