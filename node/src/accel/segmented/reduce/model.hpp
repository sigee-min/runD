#pragma once

#include <kernel/program/compute/model.hpp>

#include <cstdint>
#include <string>

namespace rund::node::accel::detail {

inline constexpr rund::kernel::u64 kSegmentedIndexWidth = 256u;
inline constexpr rund::kernel::u64 kSegmentedTeamWidth = 64u;
inline constexpr rund::kernel::u64 kSegmentedTeamsPerGroup =
    kSegmentedIndexWidth / kSegmentedTeamWidth;
inline constexpr rund::kernel::u64 kSegmentedMaxGroups = 65'535u;

inline constexpr rund::kernel::u32 kSegmentInvalid = 1u;
inline constexpr rund::kernel::u32 kSegmentSumOverflow = 2u;
inline constexpr rund::kernel::u32 kSegmentCountOverflow = 4u;

struct SegmentedReduceLayout final {
  rund::kernel::u64 block_count{};
  rund::kernel::u64 index_groups{};
};

[[nodiscard]] constexpr SegmentedReduceLayout
SegmentedReduceLayoutFor(const rund::kernel::u64 count) noexcept {
  const rund::kernel::u64 blocks =
      count / kSegmentedIndexWidth +
      (count % kSegmentedIndexWidth != 0u ? 1u : 0u);
  return {
      .block_count = blocks,
      .index_groups =
          blocks < kSegmentedMaxGroups ? blocks : kSegmentedMaxGroups,
  };
}

void AppendSegmentedReduceShaderModel(std::string &source);

} // namespace rund::node::accel::detail
