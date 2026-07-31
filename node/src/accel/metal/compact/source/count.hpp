#pragma once

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] const char* MetalCompactCountSource() {
  return R"MSL(
kernel void rund_compute_compact_count_blocks(
    device const uint* flags [[buffer(0)]],
    device uint* block_counts [[buffer(1)]],
    device uint* flag_bits [[buffer(2)]],
    constant CompactParams& params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint block [[threadgroup_position_in_grid]]) {
  threadgroup uint local[1024];
  const ulong base = ulong(block) * 1024ul;
  const ulong gid = base + ulong(tid);
  const bool active = gid < params.element_count;
  local[tid] = active && flags[gid] != 0u ? 1u : 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 32u) {
    uint word = 0u;
    for (uint bit = 0u; bit < 32u; ++bit) {
      word |= (local[tid * 32u + bit] & 1u) << bit;
    }
    flag_bits[block * 32u + tid] = word;
  }
  const uint total = rund_compact_sum_block(local, tid);
  if (tid == 0u) {
    block_counts[block] = total;
  }
}
)MSL";
}

}  // namespace
}  // namespace rund::node::accel::detail
