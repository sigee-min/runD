#pragma once

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] const char* MetalCompactScatterSource() {
  return R"MSL(
kernel void rund_compute_compact_scatter_block_offsets(
    device const uint* flag_bits [[buffer(0)]],
    device const uint* block_offsets [[buffer(1)]],
    device uint* output [[buffer(2)]],
    device const uint* scan_totals [[buffer(3)]],
    constant CompactParams& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint block [[threadgroup_position_in_grid]]) {
  threadgroup uint local[1024];
  threadgroup uint block_offset;
  if (tid == 0u) {
    block_offset = block_offsets[block];
    if (params.scan_block_size != 0u) {
      block_offset += scan_totals[block / params.scan_block_size];
    }
  }
  const ulong base = ulong(block) * 1024ul;
  const ulong gid = base + ulong(tid);
  const bool active = gid < params.element_count;
  const uint word = flag_bits[block * 32u + (tid >> 5u)];
  const uint keep = active && ((word >> (tid & 31u)) & 1u) != 0u ? 1u : 0u;
  local[tid] = keep;
  rund_compact_scan_block(local, tid);
  if (keep != 0u) {
    const uint target = block_offset + local[tid];
    if (ulong(target) < params.output_capacity) {
      output[target] = uint(gid);
    }
  }
}

kernel void rund_compute_compact_scatter(
    device const uint* flags [[buffer(0)]],
    device const uint* offsets [[buffer(1)]],
    device uint* output [[buffer(2)]],
    device const uint* scan_totals [[buffer(3)]],
    constant CompactParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint block [[threadgroup_position_in_grid]]) {
  threadgroup uint scan_offset;
  if (params.scan_block_size != 0u && tid == 0u) {
    scan_offset = scan_totals[block];
  }
  if (params.scan_block_size != 0u) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (ulong(gid) >= params.element_count || flags[gid] == 0u) {
    return;
  }
  uint target = offsets[gid];
  if (params.scan_block_size != 0u) {
    target += scan_offset;
  }
  if (ulong(target) < params.output_capacity) {
    output[target] = gid;
  }
}
)MSL";
}

}  // namespace
}  // namespace rund::node::accel::detail
