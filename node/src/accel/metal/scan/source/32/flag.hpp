#pragma once

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] const char *MetalScanBlockFlagU32Source() {
  return R"MSL(
kernel void rund_compute_scan_block_flag_u32(
    device const uint* flags [[buffer(0)]],
    device uint* output [[buffer(1)]],
    device uint* totals [[buffer(2)]],
    device uint* status [[buffer(3)]],
    constant ulong& element_count [[buffer(4)]],
    constant ulong& block_size [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]],
    uint block [[threadgroup_position_in_grid]]) {
  threadgroup ulong values[kScanWidth];
  const ulong begin = ulong(block) * block_size;
  const ulong end = min(begin + block_size, element_count);
  const ulong lane_size =
      (block_size + ulong(width) - 1ul) / ulong(width);
  const ulong lane_begin = min(begin + ulong(tid) * lane_size, end);
  const ulong lane_end = min(lane_begin + lane_size, end);
  ulong running = 0ul;
  for (ulong index = lane_begin; index < lane_end; ++index) {
    running += flags[index] == 0u ? 0ul : 1ul;
  }
  values[tid] = running;
  const ulong total = rund_scan_exclusive_ulong(values, tid, width);
  running = values[tid];
  for (ulong index = lane_begin; index < lane_end; ++index) {
    output[index] = uint(running);
    running += flags[index] == 0u ? 0ul : 1ul;
  }
  if (tid == 0u) {
    totals[block] = uint(total);
  }
  (void)status;
}
)MSL";
}

} // namespace
} // namespace rund::node::accel::detail
