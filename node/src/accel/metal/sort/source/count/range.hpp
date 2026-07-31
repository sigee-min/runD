#pragma once

namespace rund::node::accel::detail {
namespace {
[[nodiscard]] const char *MetalSortRangeSource() {
  return R"MSL(
struct SortRange {
  ulong logical;
  bool invalid;
};

inline SortRange rund_sort_range(device const uint* logical_count,
                                 constant SortParams& params) {
  ulong count = params.element_count;
  if (params.count_words == 0u) {
    return SortRange{count, false};
  }
  count = ulong(logical_count[0]);
  if (params.count_words == 2u) {
    count |= ulong(logical_count[1]) << 32u;
  }
  const bool invalid = count > params.element_count;
  return SortRange{invalid ? 0ul : count, invalid};
}
)MSL";
}
} // namespace
} // namespace rund::node::accel::detail
