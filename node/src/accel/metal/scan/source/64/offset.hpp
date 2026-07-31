#pragma once

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] const char *MetalScanOffsetU64Source() {
  return R"MSL(
kernel void rund_compute_scan_offset_u64(
    device ulong* output [[buffer(0)]],
    device const ulong* offsets [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    constant ulong& element_count [[buffer(3)]],
    constant ulong& block_size [[buffer(4)]],
    device const uint* logical_count [[buffer(5)]],
    constant uint& count_words [[buffer(6)]],
    constant uint& signed_domain [[buffer(7)]],
    device const ulong* input [[buffer(8)]],
    constant uint& inclusive [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]],
    uint block [[threadgroup_position_in_grid]]) {
  const ulong begin = ulong(block) * block_size;
  const ulong resident_count = count_words == 2u
      ? (ulong(logical_count[1]) << 32u) | ulong(logical_count[0])
      : (count_words == 1u ? ulong(logical_count[0]) : element_count);
  const ulong active_count = min(resident_count, element_count);
  const ulong end = min(begin + block_size, active_count);
  const ulong lane_size =
      (block_size + ulong(width) - 1ul) / ulong(width);
  const ulong lane_begin = min(begin + ulong(tid) * lane_size, end);
  const ulong lane_end = min(lane_begin + lane_size, end);
  const ulong offset = offsets[block];
  uint bad = 0u;
  for (ulong index = lane_begin; index < lane_end; ++index) {
    const ulong local = output[index];
    const ulong value = input[index];
    const ulong global = offset + local;
    const ulong previous = global - value;
    if (rund_scan_overflow_ulong(previous, value, global, signed_domain)) {
      bad = 1u;
    }
    output[index] = inclusive != 0u ? global : previous;
  }
  if (bad != 0u) {
    atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
  }
}
)MSL";
}

} // namespace
} // namespace rund::node::accel::detail
