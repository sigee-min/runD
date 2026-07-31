#pragma once

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] const char *MetalSegmentedBlockSource() {
  return R"MSL(
inline uint rund_segmented_overflow_u32(uint lhs, uint rhs, uint sum) {
  return sum < lhs ? 1u : 0u;
}
inline uint rund_segmented_overflow_u64(ulong lhs, ulong rhs, ulong sum) {
  return sum < lhs ? 1u : 0u;
}
inline uint rund_segmented_overflow_i32(uint lhs, uint rhs, uint sum) {
  return (((lhs ^ rhs) & 0x80000000u) == 0u &&
          ((lhs ^ sum) & 0x80000000u) != 0u) ? 1u : 0u;
}
inline uint rund_segmented_overflow_i64(ulong lhs, ulong rhs, ulong sum) {
  constexpr ulong sign = 0x8000000000000000ul;
  return (((lhs ^ rhs) & sign) == 0ul && ((lhs ^ sum) & sign) != 0ul)
             ? 1u
             : 0u;
}
#define RUND_SEGMENTED_BLOCK(SUFFIX, TYPE, ZERO, OVERFLOW) \
kernel void rund_compute_segmented_scan_block_##SUFFIX( \
    device const TYPE* input [[buffer(0)]], \
    device const uint* heads [[buffer(1)]], \
    device TYPE* output [[buffer(2)]], \
    device TYPE* offsets [[buffer(3)]], \
    device uint* first_heads [[buffer(4)]], \
    device atomic_uint* status [[buffer(5)]], \
    constant SegmentedScanParams& params [[buffer(6)]], \
    uint block [[threadgroup_position_in_grid]], \
    uint lane [[thread_position_in_threadgroup]]) { \
  if (ulong(block) >= params.block_count) { return; } \
  const ulong begin = ulong(block) * params.block_size; \
  const ulong end = min(begin + params.block_size, params.element_count); \
  const uint len = uint(end - begin); \
  threadgroup TYPE segment_values[2][256]; \
  threadgroup uint segment_flags[2][256]; \
  threadgroup TYPE segment_carry; \
  threadgroup uint segment_seen; \
  threadgroup atomic_uint segment_first; \
  threadgroup atomic_uint segment_status; \
  if (lane == 0u) { \
    segment_carry = ZERO; segment_seen = 0u; \
    atomic_store_explicit(&segment_first, len, memory_order_relaxed); \
    atomic_store_explicit(&segment_status, 0u, memory_order_relaxed); \
  } \
  threadgroup_barrier(mem_flags::mem_threadgroup); \
  for (uint tile = 0u; tile < len; tile += kSegmentedWidth) { \
    const uint local = tile + lane; \
    const bool valid = local < len; \
    const ulong index = begin + ulong(local); \
    const uint head = valid ? heads[index] : 0u; \
    uint bad = valid && (head > 1u || (index == 0ul && head != 1u)) \
                   ? 2u : 0u; \
    if (valid && head == 1u) { \
      atomic_fetch_min_explicit(&segment_first, local, memory_order_relaxed); \
    } \
    const TYPE input_value = valid ? input[index] : ZERO; \
    segment_values[0][lane] = input_value; \
    segment_flags[0][lane] = valid ? min(head, 1u) : 0u; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    uint bank = 0u; \
    for (uint step = 1u; step < kSegmentedWidth; step <<= 1u) { \
      const uint next_bank = bank ^ 1u; \
      const TYPE left = \
          lane >= step ? segment_values[bank][lane - step] : ZERO; \
      const uint left_head = \
          lane >= step ? segment_flags[bank][lane - step] : 0u; \
      TYPE value = segment_values[bank][lane]; \
      uint has_head = segment_flags[bank][lane]; \
      if (lane >= step) { \
        if (has_head == 0u) { value += left; } \
        has_head |= left_head; \
      } \
      segment_values[next_bank][lane] = value; \
      segment_flags[next_bank][lane] = has_head; \
      threadgroup_barrier(mem_flags::mem_threadgroup); \
      bank = next_bank; \
    } \
    const TYPE local_next = segment_values[bank][lane]; \
    const TYPE next = segment_flags[bank][lane] == 0u \
                          ? segment_carry + local_next : local_next; \
    const TYPE previous = head == 1u ? ZERO : next - input_value; \
    const bool owns = block == 0u || segment_seen != 0u || \
                      segment_flags[bank][lane] != 0u; \
    if (valid && owns && OVERFLOW(previous, input_value, next) != 0u) { \
      bad = max(bad, 1u); \
    } \
    if (valid) { output[index] = params.inclusive != 0u ? next : previous; } \
    if (bad != 0u) { \
      atomic_fetch_max_explicit(&segment_status, bad, memory_order_relaxed); \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0u) { \
      const uint tile_count = min(kSegmentedWidth, len - tile); \
      const uint last = tile_count - 1u; \
      segment_carry = segment_flags[bank][last] == 0u \
                          ? segment_carry + segment_values[bank][last] \
                          : segment_values[bank][last]; \
      segment_seen |= segment_flags[bank][last]; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
  } \
  if (lane == 0u) { \
    offsets[block] = segment_carry; \
    first_heads[block] = \
        atomic_load_explicit(&segment_first, memory_order_relaxed); \
    atomic_store_explicit( \
        &status[block], \
        atomic_load_explicit(&segment_status, memory_order_relaxed), \
                          memory_order_relaxed); \
  } \
}

RUND_SEGMENTED_BLOCK(u32, uint, 0u, rund_segmented_overflow_u32)
RUND_SEGMENTED_BLOCK(u64, ulong, 0ul, rund_segmented_overflow_u64)
RUND_SEGMENTED_BLOCK(i32, uint, 0u, rund_segmented_overflow_i32)
RUND_SEGMENTED_BLOCK(i64, ulong, 0ul, rund_segmented_overflow_i64)
)MSL";
}

} // namespace
} // namespace rund::node::accel::detail
