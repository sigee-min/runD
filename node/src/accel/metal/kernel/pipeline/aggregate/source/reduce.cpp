#include "reduce.hpp"

namespace rund::node::accel::detail {
namespace {

inline constexpr std::string_view MetalNestedAggregateReduce = R"rundmetal(
kernel void rund_pipeline_nested_aggregate_reduce_u32(
    device const uint *queue [[buffer(0)]],
    device const uint *domain [[buffer(1)]],
    device const uint *count [[buffer(2)]],
    constant AggregateParams &params [[buffer(3)]],
    device uint *tile_low [[buffer(4)]],
    device uint *tile_status [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]], uint group
    [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_groups [[simdgroups_per_threadgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
  threadgroup uint partial_low[32];
  threadgroup uint partial_high[32];
  threadgroup uint partial_bad[32];
  const uint outer = group;
  const uint items = count[params.count_offset_words];
  const bool count_valid = items <= params.maximum;
  const uint active_outer =
      count_valid
          ? uint((ulong(items) + ulong(params.tile) - 1ul) /
                 ulong(params.tile))
          : 0u;
  const bool active = outer < active_outer;
  // `active` is uniform for the complete threadgroup. Inactive or invalid
  // tails publish no row and may leave before any threadgroup barrier because
  // finalize derives and reads only the active prefix from the same count.
  if (!active) { return; }
  const ulong base = ulong(outer) * ulong(params.tile);
  const uint live =
      uint(min(ulong(params.tile), ulong(items) - base));
  uint2 local_sum = uint2(0u);
  uint local_bad = 0xffffffffu;
  for (ulong offset = ulong(tid); offset < ulong(live);
       offset += ulong(width)) {
    const ulong ordinal = base + offset;
    if (ordinal >= params.queue_count) {
      local_bad = min(local_bad, uint(offset));
      continue;
    }
    const uint item =
        queue[params.queue_offset_words +
              ordinal * params.queue_stride_words];
    if (ulong(item) >= params.domain_count) {
      local_bad = min(local_bad, uint(offset));
    } else {
      const uint value =
          domain[params.domain_offset_words +
                 ulong(item) * params.domain_stride_words];
      local_sum = wide_add(local_sum, uint2(value, 0u));
    }
  }
  const uint2 simd_total = simd_wide_sum(local_sum, lane, simd_width);
  const uint simd_bad = simd_min(local_bad);
  if (lane == 0u) {
    partial_low[simd_group] = simd_total.x;
    partial_high[simd_group] = simd_total.y;
    partial_bad[simd_group] = simd_bad;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0u) {
    uint2 total = lane < simd_groups
                      ? uint2(partial_low[lane], partial_high[lane])
                      : uint2(0u);
    uint bad = lane < simd_groups ? partial_bad[lane] : 0xffffffffu;
    total = simd_wide_sum(total, lane, simd_width);
    bad = simd_min(bad);
    if (lane == 0u) {
      partial_low[0] = total.x;
      partial_high[0] = total.y;
      partial_bad[0] = bad;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    const uint bad = partial_bad[0];
    tile_low[params.tile_low_offset_words + ulong(outer)] = partial_low[0];
    // Invalid offsets are strictly less than tile. `tile` is the overflow
    // marker and UINT_MAX is success; common/native admission excludes
    // tile==UINT_MAX, so all three states are disjoint in one U32 word.
    tile_status[params.tile_status_offset_words + ulong(outer)] =
        bad != 0xffffffffu
            ? bad
            : (partial_high[0] != 0u ? params.tile : 0xffffffffu);
  }
}
)rundmetal";

} // namespace

std::string_view MetalNestedAggregateReduceSource() noexcept {
  return MetalNestedAggregateReduce;
}

} // namespace rund::node::accel::detail
