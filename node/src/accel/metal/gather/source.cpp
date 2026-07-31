#include "local.hpp"

#include <string>

namespace rund::node::accel::detail {

std::string MetalGatherSource() {
  return R"MSL(
#include <metal_stdlib>
using namespace metal;

struct GatherParams {
  ulong element_count;
  ulong source_count;
  uint count_source;
  uint reserved;
};

inline ulong rund_gather_count(device const uint* words,
                               constant GatherParams& params) {
  if (params.count_source == 0u) { return params.element_count; }
  if (params.count_source == 1u) { return ulong(words[0]); }
  return ulong(words[0]) | (ulong(words[1]) << 32u);
}

kernel void rund_compute_gather_control(
    device const uint* count_words [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device uint* status [[buffer(2)]],
    device uint* indirect [[buffer(3)]],
    constant GatherParams& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]]) {
  const ulong logical = rund_gather_count(count_words, params);
  uint local_invalid = 0xffffffffu;
  if (logical <= params.element_count) {
    for (ulong ordinal = ulong(tid); ordinal < logical; ordinal += 256u) {
      if (ulong(indices[ordinal]) >= params.source_count) {
        local_invalid = min(local_invalid, uint(ordinal));
      }
    }
  }
  threadgroup uint invalids[256];
  invalids[tid] = local_invalid;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128u; stride != 0u; stride >>= 1u) {
    if (tid < stride) {
      invalids[tid] = min(invalids[tid], invalids[tid + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid != 0u) { return; }
  status[0] = 0u;
  status[1] = uint(min(logical, 0xfffffffful));
  indirect[0] = 0u;
  indirect[1] = 1u;
  indirect[2] = 1u;
  indirect[3] = 0u;
  if (logical > params.element_count) {
    status[0] = 1u;
    status[1] = uint(min(params.element_count, 0xfffffffful));
    return;
  }
  if (invalids[0] != 0xffffffffu) {
    status[0] = 2u;
    status[1] = invalids[0];
    return;
  }
  indirect[0] = uint((logical + 255u) / 256u);
  indirect[3] = uint(logical);
}

kernel void rund_compute_gather_u32(
    device const uint* values [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device uint* output [[buffer(2)]],
    device const uint* indirect [[buffer(3)]],
    constant GatherParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= indirect[3]) { return; }
  const uint source_index = indices[gid];
  output[gid] = values[source_index];
}

kernel void rund_compute_gather_u64(
    device const ulong* values [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device ulong* output [[buffer(2)]],
    device const uint* indirect [[buffer(3)]],
    constant GatherParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= indirect[3]) { return; }
  const uint source_index = indices[gid];
  output[gid] = values[source_index];
}
)MSL";
}

}  // namespace rund::node::accel::detail
