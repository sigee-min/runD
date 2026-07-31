#pragma once

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] const char *MetalScanBaseSource() {
  return R"MSL(
#include <metal_stdlib>
using namespace metal;

constant uint kScanWidth = 128u;
inline uint rund_scan_exclusive_uint(threadgroup uint* values, uint tid,
                                     uint block_size) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if ((block_size & (block_size - 1u)) == 0u) {
    for (uint stride = 1u; stride < block_size; stride <<= 1u) {
      const uint pos = (tid + 1u) * stride * 2u - 1u;
      if (pos < block_size) { values[pos] += values[pos - stride]; }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint total = values[block_size - 1u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) { values[block_size - 1u] = 0u; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = block_size >> 1u; stride > 0u; stride >>= 1u) {
      const uint pos = (tid + 1u) * stride * 2u - 1u;
      if (pos < block_size) {
        const uint left = values[pos - stride];
        values[pos - stride] = values[pos];
        values[pos] += left;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return total;
  }
  for (uint offset = 1u; offset < block_size; offset <<= 1u) {
    const uint add = tid >= offset ? values[tid - offset] : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    values[tid] += add;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint total = values[block_size - 1u];
  const uint exclusive = tid == 0u ? 0u : values[tid - 1u];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  values[tid] = exclusive;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return total;
}
inline void rund_scan_exclusive_uint_pair(
    threadgroup uint* first, threadgroup uint* second, uint tid,
    uint block_size, thread uint& exclusive, thread uint& total) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint bank = 0u;
  for (uint step = 1u; step < block_size; step <<= 1u) {
    const uint next_bank = bank ^ 1u;
    threadgroup uint* source = bank == 0u ? first : second;
    threadgroup uint* target = next_bank == 0u ? first : second;
    const uint addend = tid >= step ? source[tid - step] : 0u;
    target[tid] = source[tid] + addend;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bank = next_bank;
  }
  threadgroup uint* result = bank == 0u ? first : second;
  total = result[block_size - 1u];
  exclusive = tid == 0u ? 0u : result[tid - 1u];
}
inline ulong rund_scan_exclusive_ulong(threadgroup ulong* values, uint tid, uint block_size) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if ((block_size & (block_size - 1u)) == 0u) {
    for (uint stride = 1u; stride < block_size; stride <<= 1u) {
      const uint pos = (tid + 1u) * stride * 2u - 1u; if (pos < block_size) { values[pos] += values[pos - stride]; }
      threadgroup_barrier(mem_flags::mem_threadgroup); }
    const ulong total = values[block_size - 1u]; threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) { values[block_size - 1u] = 0ul; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = block_size >> 1u; stride > 0u; stride >>= 1u) {
      const uint pos = (tid + 1u) * stride * 2u - 1u; if (pos < block_size) {
        const ulong left = values[pos - stride]; values[pos - stride] = values[pos]; values[pos] += left; }
      threadgroup_barrier(mem_flags::mem_threadgroup); }
    return total;
  }
  for (uint offset = 1u; offset < block_size; offset <<= 1u) {
    const ulong add = tid >= offset ? values[tid - offset] : 0ul;
    threadgroup_barrier(mem_flags::mem_threadgroup); values[tid] += add; threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const ulong total = values[block_size - 1u];
  const ulong exclusive = tid == 0u ? 0ul : values[tid - 1u];
  threadgroup_barrier(mem_flags::mem_threadgroup); values[tid] = exclusive; threadgroup_barrier(mem_flags::mem_threadgroup);
  return total;
}
inline bool rund_scan_overflow_uint(uint previous, uint value, uint next,
                                    uint signed_domain) {
  if (signed_domain == 0u) { return next < previous; }
  const bool same_sign = ((previous ^ value) & 0x80000000u) == 0u;
  const bool sign_changed = ((previous ^ next) & 0x80000000u) != 0u;
  return same_sign && sign_changed;
}
inline bool rund_scan_overflow_ulong(ulong previous, ulong value, ulong next,
                                     uint signed_domain) {
  if (signed_domain == 0u) { return next < previous; }
  const bool same_sign =
      ((previous ^ value) & 0x8000000000000000ul) == 0ul;
  const bool sign_changed =
      ((previous ^ next) & 0x8000000000000000ul) != 0ul;
  return same_sign && sign_changed;
}
)MSL";
}

} // namespace
} // namespace rund::node::accel::detail
