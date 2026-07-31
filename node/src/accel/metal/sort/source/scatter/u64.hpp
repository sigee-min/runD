#pragma once

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] const char *MetalSortScatterU64Source() {
  return R"MSL(
kernel void rund_compute_sort_scatter_u64(
    device const ulong* source_keys [[buffer(0)]],
    device const uint* source_values [[buffer(1)]],
    device ulong* target_keys [[buffer(2)]],
    device uint* target_values [[buffer(3)]],
    device const uint* block_offsets [[buffer(4)]],
    device const uint* bucket_offsets [[buffer(5)]],
    constant SortParams& params [[buffer(6)]],
    device const uint* logical_count [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[simdgroup_index_in_threadgroup]],
    uint block [[threadgroup_position_in_grid]]) {
  const SortRange range = rund_sort_range(logical_count, params);
  const ulong logical = range.logical;
  const ulong begin = ulong(block) * ulong(kSortBlockSize);
  if (begin >= logical) {
    return;
  }

  threadgroup atomic_uint
      packed_counts[RUND_SORT_BUCKET_COUNT * RUND_SORT_PACKED_PAIRS];
  threadgroup uint cursors[RUND_SORT_BUCKET_COUNT * 2];
  const uint table = tid * params.block_count + block;
  cursors[tid] = bucket_offsets[tid] + block_offsets[table];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint active_count =
      uint(min(ulong(kSortBlockSize), logical - begin));
  for (uint round = 0u; round < kSortRoundCount; ++round) {
    const ulong index = begin + ulong(round * kSortThreadCount + tid);
    const bool active = index < logical;
    const ulong key = active ? source_keys[index] : 0ul;
    const uint bucket =
        active ? rund_sort_bucket_u64(key, params.pass_index, params) : 0u;
    const uint target = rund_sort_round_target(
        bucket, active, active_count, round, tid, lane, group, packed_counts,
        cursors);
    if (active) {
      target_keys[target] = key;
      target_values[target] =
          params.identity_values != 0u && params.pass_index == 0u
              ? uint(index)
              : source_values[index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
)MSL";
}

} // namespace
} // namespace rund::node::accel::detail
