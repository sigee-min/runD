#include "src/accel/segmented/reduce/status.hpp"

#include <string_view>

int RunAccelSegmentedReduceModelContract() {
  using namespace rund::node::accel::detail;
  static_assert(kSegmentedIndexWidth == 256u);
  static_assert(kSegmentedTeamWidth == 64u);
  static_assert(kSegmentedTeamsPerGroup == 4u);
  static_assert(SegmentedReduceLayoutFor(1u).block_count == 1u);
  static_assert(SegmentedReduceLayoutFor(256u).block_count == 1u);
  static_assert(SegmentedReduceLayoutFor(257u).block_count == 2u);
  static_assert(
      SegmentedReduceLayoutFor(kSegmentedIndexWidth * kSegmentedMaxGroups)
          .index_groups == kSegmentedMaxGroups);
  static_assert(SegmentedReduceLayoutFor(kSegmentedIndexWidth *
                                         (kSegmentedMaxGroups + 1u))
                    .index_groups == kSegmentedMaxGroups);
  return !SegmentedReduceStatus(0u).ok ||
                 std::string_view{SegmentedReduceStatus(kSegmentSumOverflow |
                                                        kSegmentCountOverflow)
                                      .reason} !=
                     "compute_segmented_reduce_sum_overflow" ||
                 std::string_view{SegmentedReduceStatus(kSegmentInvalid |
                                                        kSegmentSumOverflow |
                                                        kSegmentCountOverflow)
                                      .reason} !=
                     "compute_segmented_reduce_segment_invalid" ||
                 SegmentedReduceStatus(8u).ok ||
                 std::string_view{SegmentedReduceStatus(8u).reason} !=
                     "compute_segmented_reduce_invalid"
             ? 1
             : 0;
}
