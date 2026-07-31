#pragma once

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] const char* MetalScanPrefixU32Source() {
  return R"MSL(
kernel void rund_compute_scan_prefix_u32(
    device uint* totals [[buffer(0)]],
    constant ulong& block_count [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]]) {
  threadgroup uint chunks[2][kScanWidth];
  const ulong chunk_size =
      (block_count + ulong(width) - 1ul) / ulong(width);
  const ulong begin = min(ulong(tid) * chunk_size, block_count);
  const ulong end = min(begin + chunk_size, block_count);
  uint running = 0u;
  for (ulong block = begin; block < end; ++block) {
    const uint total = totals[block];
    totals[block] = running;
    running += total;
  }
  chunks[0][tid] = running;
  uint offset = 0u;
  uint total = 0u;
  rund_scan_exclusive_uint_pair(chunks[0], chunks[1], tid, width, offset,
                                total);
  for (ulong block = begin; block < end; ++block) {
    totals[block] += offset;
  }
}
)MSL";
}

}  // namespace
}  // namespace rund::node::accel::detail
