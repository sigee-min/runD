#pragma once

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] const char *MetalSortClassifySource() {
  return R"MSL(
kernel void rund_compute_sort_histogram_u32(
    device const uint* source_keys [[buffer(0)]],
    device uint* block_counts [[buffer(1)]],
    constant SortParams& params [[buffer(2)]],
    device const uint* logical_count [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[simdgroup_index_in_threadgroup]],
    uint block [[threadgroup_position_in_grid]]) {
  const ulong begin = ulong(block) * ulong(kSortBlockSize);
  const SortRange range = rund_sort_range(logical_count, params);
  const ulong logical = range.logical;
  const uint active_count =
      begin >= logical ? 0u : uint(min(ulong(kSortBlockSize), logical - begin));
  threadgroup atomic_uint counts[RUND_SORT_BUCKET_COUNT];
  atomic_store_explicit(&counts[tid], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint round = 0u; round < kSortRoundCount; ++round) {
    const ulong index = begin + ulong(round * kSortThreadCount + tid);
    const bool active = index < logical;
    const uint key = active ? source_keys[index] : 0u;
    const uint bucket =
        active ? rund_sort_bucket_u32(key, params.pass_index, params) : 0u;
    const uint equal_mask = rund_sort_equal_mask(
        bucket, rund_sort_live_mask(active_count, round, group));
    if (active && rund_sort_subgroup_rank(equal_mask, lane) == 0u) {
      atomic_fetch_add_explicit(&counts[bucket], popcount(equal_mask),
                                memory_order_relaxed);
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint target = tid * params.block_count + block;
  block_counts[target] =
      atomic_load_explicit(&counts[tid], memory_order_relaxed);
}

kernel void rund_compute_sort_histogram_u64(
    device const ulong* source_keys [[buffer(0)]],
    device uint* block_counts [[buffer(1)]],
    constant SortParams& params [[buffer(2)]],
    device const uint* logical_count [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[simdgroup_index_in_threadgroup]],
    uint block [[threadgroup_position_in_grid]]) {
  const ulong begin = ulong(block) * ulong(kSortBlockSize);
  const SortRange range = rund_sort_range(logical_count, params);
  const ulong logical = range.logical;
  const uint active_count =
      begin >= logical ? 0u : uint(min(ulong(kSortBlockSize), logical - begin));
  threadgroup atomic_uint counts[RUND_SORT_BUCKET_COUNT];
  atomic_store_explicit(&counts[tid], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint round = 0u; round < kSortRoundCount; ++round) {
    const ulong index = begin + ulong(round * kSortThreadCount + tid);
    const bool active = index < logical;
    const ulong key = active ? source_keys[index] : 0ul;
    const uint bucket =
        active ? rund_sort_bucket_u64(key, params.pass_index, params) : 0u;
    const uint equal_mask = rund_sort_equal_mask(
        bucket, rund_sort_live_mask(active_count, round, group));
    if (active && rund_sort_subgroup_rank(equal_mask, lane) == 0u) {
      atomic_fetch_add_explicit(&counts[bucket], popcount(equal_mask),
                                memory_order_relaxed);
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint target = tid * params.block_count + block;
  block_counts[target] =
      atomic_load_explicit(&counts[tid], memory_order_relaxed);
}
)MSL";
}

} // namespace
} // namespace rund::node::accel::detail
