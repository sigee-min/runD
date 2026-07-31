#include "local.hpp"
#include "source/wide.hpp"

#include <string>

namespace rund::node::accel::detail {

std::string MetalPartitionSource() {
  return std::string{R"MSL(
#include <metal_stdlib>
using namespace metal;

struct PartitionParams {
  ulong element_count;
};

kernel void rund_compute_partition_classify_u32(
    device const uint* flags [[buffer(0)]],
    device uint* false_bits [[buffer(1)]],
    constant PartitionParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (ulong(gid) >= params.element_count) { return; }
  const bool true_group = flags[gid] != 0u;
  false_bits[gid] = true_group ? 0u : 1u;
}

kernel void rund_compute_partition_scatter_u32(
    device const uint* flags [[buffer(0)]],
    device const uint* values [[buffer(1)]],
    device uint* output [[buffer(2)]],
    device const uint* false_offsets [[buffer(3)]],
    constant PartitionParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]) {
  threadgroup uint false_total_shared;
  if (lid == 0u) {
    const ulong last = params.element_count - 1ul;
    false_total_shared = false_offsets[last] +
                         (flags[last] == 0u ? 1u : 0u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (ulong(gid) >= params.element_count) { return; }
  const uint true_rank = gid - false_offsets[gid];
  const uint target = flags[gid] == 0u ? false_offsets[gid]
                                       : false_total_shared + true_rank;
  output[target] = values[gid];
}

kernel void rund_compute_partition_scatter_u64(
    device const uint* flags [[buffer(0)]],
    device const ulong* values [[buffer(1)]],
    device ulong* output [[buffer(2)]],
    device const uint* false_offsets [[buffer(3)]],
    constant PartitionParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]) {
  threadgroup uint false_total_shared;
  if (lid == 0u) {
    const ulong last = params.element_count - 1ul;
    false_total_shared = false_offsets[last] +
                         (flags[last] == 0u ? 1u : 0u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (ulong(gid) >= params.element_count) { return; }
  const uint true_rank = gid - false_offsets[gid];
  const uint target = flags[gid] == 0u ? false_offsets[gid]
                                       : false_total_shared + true_rank;
  output[target] = values[gid];
}

)MSL"} + MetalPartitionWideSource();
}

} // namespace rund::node::accel::detail
