#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::measure::compute::virtual_crossover {

// Frozen workload values are shared by the producer and packet aggregator.
// Keeping them value-only prevents either side from owning the other's logic.
inline constexpr std::size_t CorePageElements = 4'096u;
inline constexpr std::uint32_t RequestedDeviceFrames = 3u;
inline constexpr std::uint32_t RequestedHostFrames = 12u;
inline constexpr std::size_t ConditioningRuns = 60u;
inline constexpr std::array<std::size_t, 5u> LogicalCounts{
    CorePageElements - 1u, CorePageElements * 3u - 1u,
    CorePageElements * 4u - 1u, CorePageElements * 16u - 1u,
    CorePageElements * 64u - 1u};
inline constexpr std::array<std::size_t, 4u> Radii{1u, 8u, 32u, 128u};

struct ActiveRatio final {
  std::size_t numerator;
  std::size_t denominator;
};

inline constexpr std::array<ActiveRatio, 4u> ActiveRatios{
    ActiveRatio{1u, 16u}, ActiveRatio{1u, 4u}, ActiveRatio{1u, 2u},
    ActiveRatio{1u, 1u}};

inline constexpr std::size_t CellCount =
    LogicalCounts.size() * Radii.size() * ActiveRatios.size();

[[nodiscard]] constexpr std::size_t
divide_up(const std::size_t value, const std::size_t divisor) noexcept {
  return value / divisor + static_cast<std::size_t>(value % divisor != 0u);
}

static_assert(CorePageElements > Radii.back() * 2u);
static_assert(ConditioningRuns % 2u == 0u);

} // namespace rund::measure::compute::virtual_crossover
