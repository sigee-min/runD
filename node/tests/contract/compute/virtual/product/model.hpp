#pragma once

#include <cstddef>
#include <cstdint>

namespace rund_node_test_virtual::product {

constexpr std::size_t PageElements = 16u;
constexpr std::size_t ElementPageBytes = PageElements * sizeof(std::int32_t);
constexpr std::size_t ResidencyPageBytes = ElementPageBytes * 2u;
constexpr std::size_t LogicalElements = 67u;
constexpr std::size_t LogicalBytes = LogicalElements * sizeof(std::int32_t);
constexpr std::size_t FrameCapacity = 2u;
constexpr std::size_t FrameBytes = FrameCapacity * ResidencyPageBytes;
constexpr std::size_t PageCount =
    LogicalElements / PageElements +
    (LogicalElements % PageElements == 0u ? 0u : 1u);
constexpr std::size_t EpochCount =
    PageCount / FrameCapacity + (PageCount % FrameCapacity == 0u ? 0u : 1u);
constexpr std::size_t WarmRuns = 60u;
constexpr std::size_t TotalRuns = WarmRuns + 1u;
constexpr std::size_t TotalPages = TotalRuns * PageCount;
constexpr std::size_t TotalBackingReadCalls = TotalPages;
constexpr std::size_t TotalBackingWriteCalls = TotalPages;
constexpr std::size_t TotalBackingReadBytes = TotalRuns * LogicalBytes;
constexpr std::size_t TotalBackingWriteBytes = TotalRuns * LogicalBytes;
constexpr std::size_t FinalBackingReadBytes = LogicalBytes;
constexpr std::size_t FinalBackingWriteBytes = LogicalBytes;
constexpr std::size_t FinalPhysicalTransferBytes = PageCount * ElementPageBytes;
constexpr std::byte TailPoison{0xa5};
constexpr std::uint64_t GoldenHash = 0x4ce2b9f7bec321feull;

static_assert(LogicalBytes * 2u > FrameBytes);

struct BackingFacts final {
  std::uint64_t read_count{};
  std::uint64_t read_bytes{};
  std::uint64_t write_count{};
  std::uint64_t write_bytes{};
  std::uint64_t observation_count{};
  std::uint64_t observation_bytes{};
  std::uint64_t read_failure_count{};
  std::uint64_t write_failure_count{};
  std::uint64_t partial_write_bytes{};
};

} // namespace rund_node_test_virtual::product
