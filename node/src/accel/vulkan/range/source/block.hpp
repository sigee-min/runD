#pragma once

#include "../../../kernel/backend/source_recipe.hpp"
#include "algebra.hpp"

namespace rund::node::accel::detail {

template <typename Sink>
[[nodiscard]] bool EmitVulkanBlockPrefixSuffixBody(
    Sink &sink, const RangeOp op, const RangeBoundary boundary, const bool wide,
    const bool signed_values,
    const RangeExec
        &shape) noexcept(noexcept(sink.append(std::string_view{}))) {
  const char *const combine = VulkanRangeCombine(op);
  const char *const identity = VulkanRangeIdentity(op, wide, signed_values);
  if (!sink.append(R"glsl(void main() {
  const uint64_t block = uint64_t(gl_WorkGroupID.x) * uint64_t()glsl") ||
      !backend_source_recipe::append_decimal(sink, shape.width()) ||
      !sink.append(R"glsl() + uint64_t(gl_LocalInvocationID.x);
  if (params.stage == )glsl") ||
      !backend_source_recipe::append_decimal(
          sink, VulkanRangeStageValue(RangeStageKind::BlockPrefixSuffix)) ||
      !sink.append(R"glsl(u) {
    if (block >= params.stage_aux_count) { return; }
    const uint64_t window = params.window_size;
    const uint64_t begin = block * window;
    const uint64_t end = min(begin + window, params.stage_element_count);
    for (uint64_t index = begin; index < end; ++index) {
      const value_type value = index < params.padding
          ? )glsl") ||
      !sink.append(boundary == RangeBoundary::Clamp ? "input_values[0u]"
                                                    : identity) ||
      !sink.append(R"glsl(
          : (index - params.padding < params.input_count
                 ? input_values[uint(index - params.padding)]
                 : )glsl") ||
      !sink.append(boundary == RangeBoundary::Clamp
                       ? "input_values[uint(params.input_count - uint64_t(1))]"
                       : identity) ||
      !sink.append(R"glsl();
      if (index == begin) { scratch0_values[uint(index)] = value; }
      else { scratch0_values[uint(index)] = )glsl") ||
      !sink.append(combine) ||
      !sink.append(R"glsl((scratch0_values[uint(index - uint64_t(1))], value); }
    }
    for (uint64_t cursor = end; cursor > begin;) {
      const uint64_t index = cursor - uint64_t(1);
      const value_type value = index < params.padding
          ? )glsl") ||
      !sink.append(boundary == RangeBoundary::Clamp ? "input_values[0u]"
                                                    : identity) ||
      !sink.append(R"glsl(
          : (index - params.padding < params.input_count
                 ? input_values[uint(index - params.padding)]
                 : )glsl") ||
      !sink.append(boundary == RangeBoundary::Clamp
                       ? "input_values[uint(params.input_count - uint64_t(1))]"
                       : identity) ||
      !sink.append(R"glsl();
      if (index + uint64_t(1) == end) { scratch1_values[uint(index)] = value; }
      else { scratch1_values[uint(index)] = )glsl") ||
      !sink.append(combine) ||
      !sink.append(R"glsl((value, scratch1_values[uint(index + uint64_t(1))]); }
      cursor = index;
    }
    return;
  }
  if (block >= params.output_count) { return; }
  const uint64_t left = block * params.stride;
  const uint64_t right = left + params.window_size - uint64_t(1);
  output_values[uint(block)] = )glsl") ||
      !sink.append(combine) ||
      !sink.append(
          R"glsl((scratch1_values[uint(left)], scratch0_values[uint(right)]);
}
)glsl")) {
    return false;
  }
  return true;
}

} // namespace rund::node::accel::detail
