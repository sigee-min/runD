#include "model.hpp"

#include <charconv>
#include <string_view>

namespace rund::node::accel::detail {

void AppendSegmentedReduceShaderModel(std::string &source) {
  const auto append = [&source](const std::string_view name,
                                const rund::kernel::u64 value) {
    char digits[20];
    const auto result = std::to_chars(digits, digits + sizeof(digits), value);
    source.append("#define ");
    source.append(name);
    source.push_back(' ');
    source.append(digits, result.ptr);
    source.append("u\n");
  };
  append("RUND_SEGMENT_INDEX_WIDTH", kSegmentedIndexWidth);
  append("RUND_SEGMENT_TEAM_WIDTH", kSegmentedTeamWidth);
  append("RUND_SEGMENT_TEAMS_PER_GROUP", kSegmentedTeamsPerGroup);
  append("RUND_SEGMENT_MAX_GROUPS", kSegmentedMaxGroups);
  append("RUND_SEGMENT_INVALID", kSegmentInvalid);
  append("RUND_SEGMENT_SUM_OVERFLOW", kSegmentSumOverflow);
  append("RUND_SEGMENT_COUNT_OVERFLOW", kSegmentCountOverflow);
}

} // namespace rund::node::accel::detail
