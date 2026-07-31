#pragma once

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] const char *MetalSegmentedPrefixSource() {
  return R"MSL(
#define RUND_SEGMENTED_PREFIX(SUFFIX, TYPE, ZERO) \
kernel void rund_compute_segmented_scan_prefix_##SUFFIX( \
    device TYPE* offsets [[buffer(0)]], \
    device const uint* first_heads [[buffer(1)]], \
    device atomic_uint* status [[buffer(2)]], \
    constant SegmentedScanParams& params [[buffer(3)]], \
    uint lane [[thread_position_in_threadgroup]]) { \
  threadgroup TYPE segment_values[2][256]; \
  threadgroup uint segment_flags[2][256]; \
  threadgroup atomic_uint segment_status; \
  if (lane == 0u) { \
    atomic_store_explicit(&segment_status, 0u, memory_order_relaxed); \
  } \
  threadgroup_barrier(mem_flags::mem_threadgroup); \
  const ulong chunk = 1ul + \
      (params.block_count - 1ul) / ulong(kSegmentedWidth); \
  const ulong begin = min(ulong(lane) * chunk, params.block_count); \
  const ulong end = min(begin + chunk, params.block_count); \
  TYPE summary = ZERO; uint has_head = 0u; uint bad = 0u; \
  for (ulong block = begin; block < end; ++block) { \
    const ulong block_begin = block * params.block_size; \
    const ulong block_end = \
        min(block_begin + params.block_size, params.element_count); \
    const uint len = uint(block_end - block_begin); \
    const uint first = first_heads[block]; \
    const TYPE tail = offsets[block]; \
    bad = max(bad, atomic_load_explicit(&status[block], \
                                        memory_order_relaxed)); \
    const uint reset = first < len ? 1u : 0u; \
    summary = reset != 0u ? tail : summary + tail; \
    has_head |= reset; \
  } \
  segment_values[0][lane] = summary; \
  segment_flags[0][lane] = has_head; \
  if (bad != 0u) { \
    atomic_fetch_max_explicit(&segment_status, bad, memory_order_relaxed); \
  } \
  threadgroup_barrier(mem_flags::mem_threadgroup); \
  uint bank = 0u; \
  for (uint step = 1u; step < kSegmentedWidth; step <<= 1u) { \
    const uint next_bank = bank ^ 1u; \
    const TYPE left = \
        lane >= step ? segment_values[bank][lane - step] : ZERO; \
    const uint left_head = \
        lane >= step ? segment_flags[bank][lane - step] : 0u; \
    TYPE value = segment_values[bank][lane]; \
    uint reset = segment_flags[bank][lane]; \
    if (lane >= step) { \
      if (reset == 0u) { value += left; } \
      reset |= left_head; \
    } \
    segment_values[next_bank][lane] = value; \
    segment_flags[next_bank][lane] = reset; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    bank = next_bank; \
  } \
  TYPE carry = lane == 0u ? ZERO : segment_values[bank][lane - 1u]; \
  for (ulong block = begin; block < end; ++block) { \
    const ulong block_begin = block * params.block_size; \
    const ulong block_end = \
        min(block_begin + params.block_size, params.element_count); \
    const uint len = uint(block_end - block_begin); \
    const uint first = first_heads[block]; \
    const TYPE tail = offsets[block]; \
    offsets[block] = carry; \
    carry = first < len ? tail : carry + tail; \
  } \
  threadgroup_barrier(mem_flags::mem_threadgroup); \
  if (lane == 0u) { \
    atomic_store_explicit( \
        &status[0], \
        atomic_load_explicit(&segment_status, memory_order_relaxed), \
                          memory_order_relaxed); \
  } \
}

RUND_SEGMENTED_PREFIX(u32, uint, 0u)
RUND_SEGMENTED_PREFIX(u64, ulong, 0ul)
RUND_SEGMENTED_PREFIX(i32, uint, 0u)
RUND_SEGMENTED_PREFIX(i64, ulong, 0ul)
)MSL";
}

} // namespace
} // namespace rund::node::accel::detail
