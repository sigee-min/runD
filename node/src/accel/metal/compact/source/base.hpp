#pragma once

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] const char *MetalCompactBaseSource() {
  return R"MSL(
#include <metal_stdlib>
using namespace metal;

struct CompactParams {
  ulong element_count;
  ulong output_capacity;
  uint scan_block_size;
  uint reserved;
};

inline uint rund_compact_sum_block(threadgroup uint* values, uint tid) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 1u; stride < 1024u; stride <<= 1u) {
    const uint pos = (tid + 1u) * stride * 2u - 1u;
    if (pos < 1024u) {
      values[pos] += values[pos - stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return values[1023u];
}

inline uint rund_compact_scan_block(threadgroup uint* values, uint tid) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 1u; stride < 1024u; stride <<= 1u) {
    const uint pos = (tid + 1u) * stride * 2u - 1u;
    if (pos < 1024u) {
      values[pos] += values[pos - stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint total = values[1023u];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    values[1023u] = 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 512u; stride > 0u; stride >>= 1u) {
    const uint pos = (tid + 1u) * stride * 2u - 1u;
    if (pos < 1024u) {
      const uint left = values[pos - stride];
      values[pos - stride] = values[pos];
      values[pos] += left;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return total;
}
)MSL";
}

} // namespace
} // namespace rund::node::accel::detail
