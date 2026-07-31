#pragma once

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] const char* MetalScanPrefixU64Source() {
  return R"MSL(
kernel void rund_compute_scan_prefix_u64(
    device ulong* totals [[buffer(0)]],
    constant ulong& block_count [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]]) {
  threadgroup ulong chunks[kScanWidth];
  const ulong chunk_size =
      (block_count + ulong(width) - 1ul) / ulong(width);
  const ulong begin = min(ulong(tid) * chunk_size, block_count);
  const ulong end = min(begin + chunk_size, block_count);
  ulong running = 0ul;
  for (ulong block = begin; block < end; ++block) {
    const ulong total = totals[block];
    totals[block] = running;
    running += total;
  }
  chunks[tid] = running;
  rund_scan_exclusive_ulong(chunks, tid, width);
  const ulong offset = chunks[tid];
  for (ulong block = begin; block < end; ++block) {
    totals[block] += offset;
  }
}
)MSL";
}

}  // namespace
}  // namespace rund::node::accel::detail
