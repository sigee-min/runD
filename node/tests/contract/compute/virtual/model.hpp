#pragma once

#include <cstddef>
#include <cstdint>

namespace rund_node_test_virtual {

constexpr std::size_t PageElements = 16u;
constexpr std::size_t PageBytes = PageElements * sizeof(std::int32_t);
constexpr std::size_t ResidentPageCapacity = 2u;
constexpr std::size_t ResidentCapacityBytes = ResidentPageCapacity * PageBytes;
constexpr std::size_t LogicalElements = 67u;
constexpr std::size_t LogicalBytes = LogicalElements * sizeof(std::int32_t);
constexpr std::size_t BackingElements = 80u;
constexpr std::size_t BackingBytes = BackingElements * sizeof(std::int32_t);
constexpr std::size_t PageCount =
    LogicalElements / PageElements +
    (LogicalElements % PageElements == 0u ? 0u : 1u);
constexpr std::size_t WarmRunCount = 60u;
constexpr std::byte TailPoison{0xa5};
constexpr std::uint64_t GoldenHash = 0xd604a0f5e62d7bf8ull;

static_assert(LogicalBytes > ResidentCapacityBytes);
static_assert(BackingElements > LogicalElements);
static_assert(BackingBytes == PageCount * PageBytes);

struct BackingFacts final {
  std::uint64_t load_count{};
  std::uint64_t load_bytes{};
  std::uint64_t store_count{};
  std::uint64_t store_bytes{};
  std::uint64_t observation_count{};
  std::uint64_t observation_bytes{};
};

struct PagerFacts final {
  std::uint64_t run_count{};
  std::uint64_t epoch_count{};
  std::uint64_t resident_capacity_bytes{};
  std::uint64_t resident_current_bytes{};
  std::uint64_t resident_peak_bytes{};
};

} // namespace rund_node_test_virtual
