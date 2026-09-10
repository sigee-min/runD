#pragma once

#include "../../../range_aggregate/execution/projection.hpp"

#include <string_view>

namespace rund::node::accel::detail {

[[nodiscard]] constexpr std::uint32_t
MetalRangeStageValue(const RangeStageKind stage) noexcept {
  return static_cast<std::uint32_t>(stage);
}

template <typename Sink>
inline void AppendMetalPrefixDifferenceKernel(Sink &source,
                                              const RangeBoundary boundary,
                                              const RangeExec &shape,
                                              const char *const type,
                                              const char *const suffix) {
  source += "kernel void rund_range_sum_";
  source += suffix;
  source += R"MSL((
    device const )MSL";
  source += type;
  source += R"MSL(* input [[buffer(0)]],
    device )MSL";
  source += type;
  source += R"MSL(* output [[buffer(1)]],
    constant RangeParams& params [[buffer(2)]],
    device )MSL";
  source += type;
  source += R"MSL(* scratch0 [[buffer(3)]],
    device )MSL";
  source += type;
  source += R"MSL(* scratch1 [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint physical_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
  const uint lane = simd_prefix_exclusive_sum(1u);
  const uint tid = simd_group * simd_width + lane;
  threadgroup )MSL";
  source += type;
  source += " scan[";
  (void)source.decimal(shape.width());
  source += R"MSL(];
  const ulong group_base = ulong(group) * )MSL";
  (void)source.decimal(shape.width());
  source += R"MSL(ul;
  if (params.stage == )MSL";
  (void)source.decimal(MetalRangeStageValue(RangeStageKind::PrefixBlock));
  source += R"MSL(u || params.stage == )MSL";
  (void)source.decimal(MetalRangeStageValue(RangeStageKind::PrefixSummary));
  source += R"MSL(u) {
    const ulong i = group_base + ulong(tid);
    const bool active = i < params.stage_element_count;
    const )MSL";
  source += type;
  source += R"MSL( value = active
        ? (params.stage == )MSL";
  (void)source.decimal(MetalRangeStageValue(RangeStageKind::PrefixBlock));
  source += R"MSL(u ? input[i] : scratch0[i])
        : )MSL";
  source += type;
  source += R"MSL((0);
)MSL";
  if (std::string_view{type} == "uint") {
    source +=
        R"MSL(    const uint local_prefix = simd_prefix_inclusive_sum(value);
    const uint local_total = simd_sum(value);
    if (lane == 0u) { scan[simd_group] = local_total; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      uint offset = 0u;
      const uint simd_groups = ()MSL";
    (void)source.decimal(shape.width());
    source += R"MSL(u + simd_width - 1u) / simd_width;
      for (uint index = 0u; index < simd_groups; ++index) {
        const uint total = scan[index];
        scan[index] = offset;
        offset += total;
      }
      if (params.stage_aux_count > 1ul) { scratch1[group] = offset; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) { scratch0[i] = scan[simd_group] + local_prefix; }
    return;
)MSL";
  } else {
    source +=
        R"MSL(    const ulong local_prefix = rund_simd_prefix_u64(value);
    const uint last = simd_max(physical_lane);
    const ulong local_total = rund_simd_last_u64(local_prefix, last);
    if (lane == 0u) { scan[simd_group] = local_total; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      ulong offset = 0ul;
      const uint simd_groups = ()MSL";
    (void)source.decimal(shape.width());
    source += R"MSL(u + simd_width - 1u) / simd_width;
      for (uint index = 0u; index < simd_groups; ++index) {
        const ulong total = scan[index];
        scan[index] = offset;
        offset += total;
      }
      if (params.stage_aux_count > 1ul) { scratch1[group] = offset; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) { scratch0[i] = scan[simd_group] + local_prefix; }
    return;
)MSL";
  }
  source += R"MSL(  }
  if (params.stage == )MSL";
  (void)source.decimal(MetalRangeStageValue(RangeStageKind::PrefixFixup));
  source += R"MSL(u) {
    const ulong i = group_base + ulong(tid);
    if (i < params.stage_element_count && group != 0u) {
      scratch0[i] += scratch1[group - 1u];
    }
    return;
  }
  const ulong i = group_base + ulong(tid);
  if (i >= params.output_count) { return; }
  const ulong anchor = i * params.stride;
  const ulong left = anchor < params.padding ? 0ul : anchor - params.padding;
  const ulong right_width = params.window_size - params.padding;
  const ulong right =
      anchor >= params.input_count
          ? params.input_count - 1ul
          : (right_width >= params.input_count - anchor
                 ? params.input_count - 1ul
                 : anchor + right_width - 1ul);
  )MSL";
  source += type;
  source += R"MSL( value = scratch0[right];
  const ulong right_group = right / )MSL";
  (void)source.decimal(shape.width());
  source += R"MSL(ul;
  if (right_group != 0ul) { value += scratch1[right_group - 1ul]; }
  if (left != 0ul) {
    value -= scratch0[left - 1ul];
    const ulong left_group = (left - 1ul) / )MSL";
  (void)source.decimal(shape.width());
  source += R"MSL(ul;
    if (left_group != 0ul) { value -= scratch1[left_group - 1ul]; }
  }
  )MSL";
  if (boundary == RangeBoundary::Clamp) {
    source += R"MSL(  const ulong left_missing =
      anchor < params.padding ? params.padding - anchor : 0ul;
  const ulong right_missing =
      anchor >= params.input_count
          ? anchor - params.input_count + right_width
          : (right_width > params.input_count - anchor
                 ? right_width - (params.input_count - anchor)
                 : 0ul);
  if (left_missing != 0ul) { value += )MSL";
    source += type;
    source += R"MSL((left_missing * input[0]); }
  if (right_missing != 0ul) { value += )MSL";
    source += type;
    source += R"MSL((right_missing * input[params.input_count - 1ul]); }
)MSL";
  }
  source += R"MSL(
  output[i] = value;
}
)MSL";
}

} // namespace rund::node::accel::detail
