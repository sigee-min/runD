#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::measure::compute::virtual_residency {

inline constexpr std::size_t PageElements = 4'096u;
inline constexpr std::size_t LogicalElements = 65'537u;
inline constexpr std::uint32_t FrameCapacity = 3u;
inline constexpr std::size_t WarmSamples = 60u;
inline constexpr std::byte TailPoison{0xa5};
inline constexpr std::array<std::size_t, 4u> ActiveCounts{
    0u, 17u, LogicalElements / 2u + 1u, LogicalElements};
inline constexpr std::size_t FrameElements = PageElements * FrameCapacity;
inline constexpr std::size_t ResidentBytes =
    FrameElements * sizeof(std::int32_t) * 4u;
inline constexpr std::size_t HostStagingBytes =
    FrameElements * sizeof(std::int32_t) * 4u;
inline constexpr std::size_t PageCount =
    LogicalElements / PageElements +
    (LogicalElements % PageElements == 0u ? 0u : 1u);
inline constexpr std::size_t EpochCount =
    PageCount / FrameCapacity + (PageCount % FrameCapacity == 0u ? 0u : 1u);

static_assert(LogicalElements > FrameElements);
static_assert(PageCount == 17u);
static_assert(EpochCount == 6u);

[[nodiscard]] constexpr std::size_t
nearest_rank_index(const std::size_t count, const std::size_t percentile) {
  return (count * percentile + 99u) / 100u - 1u;
}

static_assert(nearest_rank_index(WarmSamples, 95u) == 56u);

struct WallSamples final {
  std::array<double, WarmSamples> microseconds{};

  void sort() noexcept;
  [[nodiscard]] double p50() const noexcept;
  [[nodiscard]] double p95() const noexcept;
};

// User wall-clock ownership is intentionally separate from Profile. These
// spans enclose public API calls; every backend, transfer, allocation and
// memory field is copied from Profile by the report leaf.
struct WallEvidence final {
  double first_result_us{};
  double warm_p50_us{};
  double warm_p95_us{};
  double logical_elements_per_s{};
  double author_us{};
  double compile_us{};
  double prepare_us{};
  double seed_us{};
  double run_us{};
  double read_us{};
};

// Harness-owned observation evidence. These counts describe explicit public
// terminal operations and are never projected as backend counters.
struct ObservationEvidence final {
  std::uint32_t terminal_reads{};
  std::uint32_t profile_projections{};

  [[nodiscard]] constexpr bool exact_terminal_only() const noexcept {
    return terminal_reads == 1u && profile_projections == 1u;
  }
};

} // namespace rund::measure::compute::virtual_residency
