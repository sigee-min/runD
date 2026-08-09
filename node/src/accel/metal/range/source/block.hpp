#pragma once

#include "algebra.hpp"

namespace rund::node::accel::detail {

template <typename Sink>
inline void AppendMetalBlockPrefixSuffixKernel(Sink &source, const RangeOp op,
                                               const RangeBoundary boundary,
                                               const RangeExec &shape,
                                               const char *const type,
                                               const char *const suffix,
                                               const char *const identity) {
  source += "kernel void rund_range_";
  source += MetalRangeOpName(op);
  source += "_";
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
  source += R"MSL(* forward_values [[buffer(3)]],
    device )MSL";
  source += type;
  source += R"MSL(* backward_values [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  const ulong base = ulong(group) * )MSL";
  (void)source.decimal(shape.width());
  source += R"MSL(ul + ulong(tid);
  if (params.stage == )MSL";
  (void)source.decimal(
      static_cast<std::uint32_t>(RangeStageKind::BlockPrefixSuffix));
  source += R"MSL(u) {
    if (base >= params.stage_aux_count) { return; }
    const ulong window = params.window_size;
    const ulong begin = base * window;
    const ulong end = min(begin + window, params.stage_element_count);
    for (ulong index = begin; index < end; ++index) {
      const )MSL";
  source += type;
  source += R"MSL( value = index < params.padding
          ? )MSL";
  source += boundary == RangeBoundary::Clamp ? "input[0]" : identity;
  source += R"MSL(
          : (index - params.padding < params.input_count
                 ? input[index - params.padding]
                 : )MSL";
  source += boundary == RangeBoundary::Clamp ? "input[params.input_count - 1ul]"
                                             : identity;
  source += R"MSL();
      if (index == begin) { forward_values[index] = value; }
      else { forward_values[index] = )MSL";
  source += op == RangeOp::Minimum ? "min" : "max";
  source += R"MSL((forward_values[index - 1ul], value); }
    }
    for (ulong cursor = end; cursor > begin;) {
      const ulong index = cursor - 1ul;
      const )MSL";
  source += type;
  source += R"MSL( value = index < params.padding
          ? )MSL";
  source += boundary == RangeBoundary::Clamp ? "input[0]" : identity;
  source += R"MSL(
          : (index - params.padding < params.input_count
                 ? input[index - params.padding]
                 : )MSL";
  source += boundary == RangeBoundary::Clamp ? "input[params.input_count - 1ul]"
                                             : identity;
  source += R"MSL();
      if (index + 1ul == end) { backward_values[index] = value; }
      else { backward_values[index] = )MSL";
  source += op == RangeOp::Minimum ? "min" : "max";
  source += R"MSL((value, backward_values[index + 1ul]); }
      cursor = index;
    }
    return;
  }
  const ulong i = base;
  if (i >= params.output_count) { return; }
  const ulong left = i * params.stride;
  const ulong right = left + params.window_size - 1ul;
  output[i] = )MSL";
  source += op == RangeOp::Minimum ? "min" : "max";
  source += R"MSL((backward_values[left], forward_values[right]);
}
)MSL";
}

} // namespace rund::node::accel::detail
