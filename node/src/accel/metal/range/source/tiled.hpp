#pragma once
#include "../../../range_aggregate/execution/projection.hpp"
#include <string_view>
namespace rund::node::accel::detail {
template <typename Sink>
void AppendMetalTiledDifferenceKernel(Sink &source, const RangeExec &shape,
                                      const char *type, const char *suffix) {
  source += R"RANGE(kernel void rund_range_sum_)RANGE";
  source += suffix;
  source += R"RANGE((
    device const )RANGE";
  source += type;
  source += R"RANGE(* input [[buffer(0)]],
    device )RANGE";
  source += type;
  source += R"RANGE(* output [[buffer(1)]],
    constant RangeParams& params [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint physical_lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint sw [[threads_per_simdgroup]]) {
  const uint lane = simd_prefix_exclusive_sum(1u);
  const uint tid = sg * sw + lane;
  constexpr uint W = )RANGE";
  (void)source.decimal(shape.width());
  source += R"RANGE(u, R = )RANGE";
  (void)source.decimal(kRangeTileOutputsPerLane);
  source += R"RANGE(u;
  threadgroup )RANGE";
  source += type;
  source += R"RANGE( deltas[W];
  threadgroup )RANGE";
  source += type;
  source += R"RANGE( anchors[W];
  const ulong base = ulong(group) * ulong(W * R);
  if (base >= params.input_count) { return; }
  const ulong radius = params.padding;
  const ulong left = base < radius ? 0ul : base - radius;
  const ulong right = radius >= params.input_count - base
      ? params.input_count - 1ul : base + radius;
  )RANGE";
  source += type;
  source += R"RANGE( anchor_sum = 0;
  for (ulong offset = tid; offset < right - left + 1ul; offset += W) {
    anchor_sum += input[left + offset];
  }
  if (tid == 0u) {
    if (base < radius) { anchor_sum += )RANGE";
  source += type;
  source += R"RANGE(((radius - base) * input[0]); }
    if (radius >= params.input_count - base) {
      anchor_sum += )RANGE";
  source += type;
  source += R"RANGE(((radius - (params.input_count - base) + 1ul) *
          input[params.input_count - 1ul]);
    }
  }
  )RANGE";
  source += type;
  source += R"RANGE( prefixes[R];
  )RANGE";
  source += type;
  source += R"RANGE( total = 0;
  for (uint j = 0u; j < R; ++j) {
    const ulong i = base + ulong(tid) * R + j;
    if (i < params.input_count && i != base) {
      const ulong right_index = radius >= params.input_count - i
          ? params.input_count - 1ul : i + radius;
      const ulong left_index = i <= radius ? 0ul : i - radius - 1ul;
      total += input[right_index] - input[left_index];
    }
    prefixes[j] = total;
  }
)RANGE";
  if (std::string_view{type} == "uint") {
    source +=
        R"RANGE(  const uint thread_offset = simd_prefix_exclusive_sum(total);
  const uint group_delta = simd_sum(total);
  const uint group_anchor = simd_sum(anchor_sum);
  if (lane == 0u) { deltas[sg] = group_delta; anchors[sg] = group_anchor; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    uint prior = 0u, initial = 0u;
    for (uint s = 0u; s < (W + sw - 1u) / sw; ++s) {
      const uint value = deltas[s];
      deltas[s] = prior;
      prior += value;
      initial += anchors[s];
    }
    anchors[0] = initial;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint start = anchors[0] + deltas[sg] + thread_offset;
)RANGE";
  } else {
    source +=
        R"RANGE(  const uint last = simd_max(physical_lane);
  const ulong inclusive = rund_simd_prefix_u64(total);
  const ulong thread_offset = inclusive - total;
  const ulong group_delta = rund_simd_last_u64(inclusive, last);
  const ulong group_anchor = rund_simd_last_u64(
      rund_simd_prefix_u64(anchor_sum), last);
  if (lane == 0u) { deltas[sg] = group_delta; anchors[sg] = group_anchor; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    ulong prior = 0ul, initial = 0ul;
    for (uint s = 0u; s < (W + sw - 1u) / sw; ++s) {
      const ulong value = deltas[s];
      deltas[s] = prior;
      prior += value;
      initial += anchors[s];
    }
    anchors[0] = initial;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const ulong start = anchors[0] + deltas[sg] + thread_offset;
)RANGE";
  }
  source += R"RANGE(  for (uint j = 0u; j < R; ++j) {
    const ulong i = base + ulong(tid) * R + j;
    if (i < params.output_count) { output[i] = start + prefixes[j]; }
  }
}
)RANGE";
}
} // namespace rund::node::accel::detail
