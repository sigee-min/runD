#pragma once

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] const char *MetalSortPrefixSource() {
  return R"MSL(
kernel void rund_compute_sort_prefix(
    device const uint* block_counts [[buffer(0)]],
    device uint* block_offsets [[buffer(1)]],
    device uint* bucket_totals [[buffer(2)]],
    constant SortParams& params [[buffer(3)]],
    device const uint* logical_count [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint bucket [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint group_count [[simdgroups_per_threadgroup]]) {
  const SortRange range = rund_sort_range(logical_count, params);
  const uint active_blocks = uint(
      (range.logical + ulong(kSortBlockSize) - 1ul) / ulong(kSortBlockSize));
  const uint chunk =
      (active_blocks + kSortThreadCount - 1u) / kSortThreadCount;
  const uint begin = tid * chunk;
  const uint end = min(begin + chunk, active_blocks);
  const uint table = bucket * params.block_count;
  uint local = 0u;
  for (uint block = begin; block < end; ++block) {
    block_offsets[table + block] = local;
    local += block_counts[table + block];
  }
  threadgroup uint group_sums[32];
  const uint offset = rund_sort_exclusive_prefix(
      local, group_sums, lane, group, simd_width, group_count);
  for (uint block = begin; block < end; ++block) {
    block_offsets[table + block] += offset;
  }
  if (tid + 1u == kSortThreadCount) {
    bucket_totals[bucket] = offset + local;
  }
}

kernel void rund_compute_sort_base(
    device const uint* bucket_totals [[buffer(0)]],
    device uint* bucket_offsets [[buffer(1)]],
    constant SortParams& params [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint group_count [[simdgroups_per_threadgroup]]) {
  const uint total = bucket_totals[tid];
  threadgroup uint group_sums[32];
  bucket_offsets[tid] = rund_sort_exclusive_prefix(
      total, group_sums, lane, group, simd_width, group_count);
}
)MSL";
}

} // namespace
} // namespace rund::node::accel::detail
