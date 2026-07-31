#pragma once

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] const char *MetalSortBlockRankSource() {
  return R"MSL(
inline uint rund_sort_live_mask(
    uint active_count,
    uint round,
    uint group) {
  const uint begin = round * kSortThreadCount + group * kSortSimdWidth;
  if (begin >= active_count) {
    return 0u;
  }
  const uint count = min(active_count - begin, kSortSimdWidth);
  return count == kSortSimdWidth ? 0xffffffffu : ((1u << count) - 1u);
}

inline uint rund_sort_equal_mask(uint bucket, uint live_mask) {
  const uint bit0 = uint((simd_vote::vote_t)simd_ballot((bucket & 1u) != 0u));
  const uint bit1 = uint((simd_vote::vote_t)simd_ballot((bucket & 2u) != 0u));
  const uint bit2 = uint((simd_vote::vote_t)simd_ballot((bucket & 4u) != 0u));
  const uint bit3 = uint((simd_vote::vote_t)simd_ballot((bucket & 8u) != 0u));
  const uint bit4 = uint((simd_vote::vote_t)simd_ballot((bucket & 16u) != 0u));
  const uint bit5 = uint((simd_vote::vote_t)simd_ballot((bucket & 32u) != 0u));
  const uint bit6 = uint((simd_vote::vote_t)simd_ballot((bucket & 64u) != 0u));
  const uint bit7 = uint((simd_vote::vote_t)simd_ballot((bucket & 128u) != 0u));
  uint equal = live_mask;
  equal &= (bucket & 1u) != 0u ? bit0 : ~bit0;
  equal &= (bucket & 2u) != 0u ? bit1 : ~bit1;
  equal &= (bucket & 4u) != 0u ? bit2 : ~bit2;
  equal &= (bucket & 8u) != 0u ? bit3 : ~bit3;
  equal &= (bucket & 16u) != 0u ? bit4 : ~bit4;
  equal &= (bucket & 32u) != 0u ? bit5 : ~bit5;
  equal &= (bucket & 64u) != 0u ? bit6 : ~bit6;
  equal &= (bucket & 128u) != 0u ? bit7 : ~bit7;
  return equal;
}

inline uint rund_sort_subgroup_rank(uint equal_mask, uint lane) {
  const uint prior = lane == 0u ? 0u : ((1u << lane) - 1u);
  return popcount(equal_mask & prior);
}

inline uint rund_sort_round_target(
    uint bucket,
    bool active,
    uint active_count,
    uint round,
    uint tid,
    uint lane,
    uint group,
    threadgroup atomic_uint* packed_counts,
    threadgroup uint* cursors) {
  for (uint index = tid; index < kSortPackedEntryCount;
       index += kSortThreadCount) {
    atomic_store_explicit(&packed_counts[index], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint live_mask = rund_sort_live_mask(active_count, round, group);
  const uint equal_mask = rund_sort_equal_mask(bucket, live_mask);
  const uint subgroup_rank = rund_sort_subgroup_rank(equal_mask, lane);
  if (active && subgroup_rank == 0u) {
    const uint word = bucket * kSortPackedPairs + group / 2u;
    const uint shift = (group & 1u) * 16u;
    const uint count = popcount(equal_mask);
    atomic_fetch_or_explicit(&packed_counts[word], count << shift,
                             memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < kSortBucketCount) {
    uint prefix = 0u;
    for (uint pair = 0u; pair < kSortPackedPairs; ++pair) {
      const uint word = tid * kSortPackedPairs + pair;
      const uint counts =
          atomic_load_explicit(&packed_counts[word], memory_order_relaxed);
      const uint low_count = counts & 0xffffu;
      const uint low_prefix = prefix;
      prefix += low_count;
      const uint high_prefix = prefix;
      prefix += counts >> 16u;
      atomic_store_explicit(
          &packed_counts[word], low_prefix | (high_prefix << 16u),
          memory_order_relaxed);
    }
    const uint current_bank = (round & 1u) * kSortBucketCount;
    const uint next_bank = ((round + 1u) & 1u) * kSortBucketCount;
    cursors[next_bank + tid] = cursors[current_bank + tid] + prefix;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (!active) {
    return 0u;
  }
  const uint word = bucket * kSortPackedPairs + group / 2u;
  const uint shift = (group & 1u) * 16u;
  const uint group_prefix =
      (atomic_load_explicit(&packed_counts[word], memory_order_relaxed) >>
       shift) &
      0xffffu;
  const uint current_bank = (round & 1u) * kSortBucketCount;
  return cursors[current_bank + bucket] + group_prefix + subgroup_rank;
}
)MSL";
}

} // namespace
} // namespace rund::node::accel::detail
