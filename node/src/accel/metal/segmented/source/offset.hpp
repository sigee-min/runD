#pragma once

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] const char *MetalSegmentedOffsetSource() {
  return R"MSL(
#define RUND_SEGMENTED_OFFSET(SUFFIX, TYPE, OVERFLOW) \
kernel void rund_compute_segmented_scan_offset_##SUFFIX( \
    device const TYPE* input [[buffer(0)]], \
    device TYPE* output [[buffer(1)]], \
    device const TYPE* offsets [[buffer(2)]], \
    device const uint* first_heads [[buffer(3)]], \
    device atomic_uint* status [[buffer(4)]], \
    constant SegmentedScanParams& params [[buffer(5)]], \
    uint block [[threadgroup_position_in_grid]], \
    uint lane [[thread_position_in_threadgroup]]) { \
  if (block == 0u || ulong(block) >= params.block_count) { return; } \
  const ulong begin = ulong(block) * params.block_size; \
  const ulong end = min(begin + params.block_size, params.element_count); \
  const uint stop = min(first_heads[block], uint(end - begin)); \
  const TYPE carry = offsets[block]; uint bad = 0u; \
  for (uint local = lane; local < stop; local += kSegmentedWidth) { \
    const ulong index = begin + ulong(local); \
    const TYPE value = input[index]; \
    const TYPE local_result = output[index]; \
    const TYPE local_previous = \
        params.inclusive != 0u ? local_result - value : local_result; \
    const TYPE previous = carry + local_previous; \
    const TYPE next = previous + value; \
    bad = max(bad, OVERFLOW(previous, value, next)); \
    output[index] = params.inclusive != 0u ? next : previous; \
  } \
  if (bad != 0u) { \
    atomic_fetch_max_explicit(&status[0], bad, memory_order_relaxed); \
  } \
}

RUND_SEGMENTED_OFFSET(u32, uint, rund_segmented_overflow_u32)
RUND_SEGMENTED_OFFSET(u64, ulong, rund_segmented_overflow_u64)
RUND_SEGMENTED_OFFSET(i32, uint, rund_segmented_overflow_i32)
RUND_SEGMENTED_OFFSET(i64, ulong, rund_segmented_overflow_i64)
)MSL";
}

} // namespace
} // namespace rund::node::accel::detail
