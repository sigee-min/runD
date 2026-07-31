#pragma once
namespace rund::node::accel::detail {
namespace {
[[nodiscard]] const char *MetalSortBaseSource() {
  return R"MSL(
constant uint kSortBlockSize = RUND_SORT_BLOCK_SIZE;
constant uint kSortThreadCount = RUND_SORT_THREAD_COUNT;
constant uint kSortRoundCount = RUND_SORT_ROUND_COUNT;
constant uint kSortBucketCount = RUND_SORT_BUCKET_COUNT;
constant uint kSortSimdWidth = RUND_SORT_SIMD_WIDTH;
constant uint kSortPackedPairs = RUND_SORT_PACKED_PAIRS;
constant uint kSortPackedEntryCount =
    RUND_SORT_BUCKET_COUNT * RUND_SORT_PACKED_PAIRS;

struct SortParams {
  ulong element_count;
  uint block_count;
  uint pass_index;
  uint identity_values;
  uint signed_order;
  uint pass_count;
  uint count_words;
};

inline uint rund_sort_bucket_u32(uint key, uint pass,
                                 constant SortParams& params) {
  const uint shift = pass * 8u;
  uint bucket = (key >> shift) & 255u;
  if (params.signed_order != 0u &&
      pass + 1u == params.pass_count) { bucket ^= 128u; }
  return bucket;
}

inline uint rund_sort_bucket_u64(ulong key, uint pass,
                                 constant SortParams& params) {
  const uint shift = pass * 8u;
  uint bucket = uint((key >> shift) & 255ul);
  if (params.signed_order != 0u &&
      pass + 1u == params.pass_count) { bucket ^= 128u; }
  return bucket;
}

inline uint rund_sort_exclusive_prefix(
    uint value,
    threadgroup uint* group_sums,
    uint lane,
    uint group,
    uint simd_width,
    uint group_count) {
  const uint lane_prefix = simd_prefix_exclusive_sum(value);
  const uint lane_total = simd_sum(value);
  if (lane == 0u) { group_sums[group] = lane_total; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (group == 0u) {
    const uint chunk = (group_count + simd_width - 1u) / simd_width;
    const uint begin = lane * chunk;
    const uint end = min(begin + chunk, group_count);
    uint local = 0u;
    for (uint index = begin; index < end; ++index) {
      local += group_sums[index];
    }
    uint offset = simd_prefix_exclusive_sum(local);
    for (uint index = begin; index < end; ++index) {
      const uint current = group_sums[index];
      group_sums[index] = offset;
      offset += current;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return group_sums[group] + lane_prefix;
}
)MSL";
}

} // namespace
} // namespace rund::node::accel::detail
