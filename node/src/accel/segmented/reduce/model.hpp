#pragma once

#include <kernel/program/compute/model.hpp>

#include "../../kernel/backend/source_recipe.hpp"

#include <cstdint>
#include <string>
#include <string_view>

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

template <typename Sink>
[[nodiscard]] bool AppendSegmentedReduceShaderModel(Sink &sink) noexcept {
  const auto append = [&sink](const std::string_view name,
                              const rund::kernel::u64 value) noexcept {
    return sink.append("#define ") && sink.append(name) && sink.append(" ") &&
           backend_source_recipe::append_decimal(sink, value) &&
           sink.append("u\n");
  };
  return append("RUND_SEGMENT_INDEX_WIDTH", kSegmentedIndexWidth) &&
         append("RUND_SEGMENT_TEAM_WIDTH", kSegmentedTeamWidth) &&
         append("RUND_SEGMENT_TEAMS_PER_GROUP", kSegmentedTeamsPerGroup) &&
         append("RUND_SEGMENT_MAX_GROUPS", kSegmentedMaxGroups) &&
         append("RUND_SEGMENT_INVALID", kSegmentInvalid) &&
         append("RUND_SEGMENT_SUM_OVERFLOW", kSegmentSumOverflow) &&
         append("RUND_SEGMENT_COUNT_OVERFLOW", kSegmentCountOverflow);
}

} // namespace rund::node::accel::detail
