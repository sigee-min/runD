#pragma once

#include "../local.hpp"

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
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
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
    scan[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 1u; offset < )MSL";
  (void)source.decimal(shape.width());
  source += R"MSL(u; offset <<= 1u) {
      const uint tree = (tid + 1u) * offset * 2u - 1u;
      if (tree < )MSL";
  (void)source.decimal(shape.width());
  source += R"MSL(u) { scan[tree] += scan[tree - offset]; }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
      if (params.stage_aux_count > 1ul) { scratch1[group] = scan[)MSL";
  (void)source.decimal(shape.width() - 1u);
  source += R"MSL(]; }
      scan[)MSL";
  (void)source.decimal(shape.width() - 1u);
  source += R"MSL(] = )MSL";
  source += type;
  source += R"MSL((0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = )MSL";
  (void)source.decimal(shape.width() / 2u);
  source += R"MSL(u; offset > 0u; offset >>= 1u) {
      const uint tree = (tid + 1u) * offset * 2u - 1u;
      if (tree < )MSL";
  (void)source.decimal(shape.width());
  source += R"MSL(u) {
        const )MSL";
  source += type;
  source += R"MSL( prior = scan[tree - offset];
        scan[tree - offset] = scan[tree];
        scan[tree] += prior;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) { scratch0[i] = scan[tid] + value; }
    return;
  }
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
  if (left != 0ul) { value -= scratch0[left - 1ul]; }
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
