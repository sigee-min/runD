#pragma once

#include "../../../kernel/backend/source_recipe.hpp"
#include "algebra.hpp"

namespace rund::node::accel::detail {

template <typename Sink>
[[nodiscard]] bool EmitVulkanPrefixDifferenceBody(
    Sink &sink, const bool wide, const RangeBoundary boundary,
    const RangeExec
        &shape) noexcept(noexcept(sink.append(std::string_view{}))) {
  const char *const scalar = wide ? "uint64_t" : "uint";
  if (!sink.append(R"glsl(void main() {
  const uint lane = gl_LocalInvocationID.x;
  const uint64_t group_base = uint64_t(gl_WorkGroupID.x) * uint64_t()glsl") ||
      !backend_source_recipe::append_decimal(sink, shape.width()) ||
      !sink.append(R"glsl();
  if (params.stage == )glsl") ||
      !backend_source_recipe::append_decimal(
          sink, VulkanRangeStageValue(RangeStageKind::PrefixBlock)) ||
      !sink.append(R"glsl(u || params.stage == )glsl") ||
      !backend_source_recipe::append_decimal(
          sink, VulkanRangeStageValue(RangeStageKind::PrefixSummary)) ||
      !sink.append(R"glsl(u) {
    const uint64_t index = group_base + uint64_t(lane);
    const bool is_active = index < params.stage_element_count;
    const )glsl") ||
      !sink.append(scalar) || !sink.append(R"glsl( value = is_active
        ? (params.stage == )glsl") ||
      !backend_source_recipe::append_decimal(
          sink, VulkanRangeStageValue(RangeStageKind::PrefixBlock)) ||
      !sink.append(R"glsl(u ? input_values[uint(index)]
                            : scratch0_values[uint(index)])
        : )glsl") ||
      !sink.append(scalar) || !sink.append(R"glsl((0);
    range_scan[lane] = value;
    barrier();
    for (uint offset = 1u; offset < )glsl") ||
      !backend_source_recipe::append_decimal(sink, shape.width()) ||
      !sink.append(R"glsl(u; offset <<= 1u) {
      const uint tree = (lane + 1u) * offset * 2u - 1u;
      if (tree < )glsl") ||
      !backend_source_recipe::append_decimal(sink, shape.width()) ||
      !sink.append(R"glsl(u) { range_scan[tree] += range_scan[tree - offset]; }
      barrier();
    }
    if (lane == 0u) {
      if (params.stage_aux_count > uint64_t(1)) {
        scratch1_values[gl_WorkGroupID.x] = range_scan[)glsl") ||
      !backend_source_recipe::append_decimal(sink, shape.width() - 1u) ||
      !sink.append(R"glsl(];
      }
      range_scan[)glsl") ||
      !backend_source_recipe::append_decimal(sink, shape.width() - 1u) ||
      !sink.append(R"glsl(] = )glsl") || !sink.append(scalar) ||
      !sink.append(R"glsl((0);
    }
    barrier();
    for (uint offset = )glsl") ||
      !backend_source_recipe::append_decimal(sink, shape.width() / 2u) ||
      !sink.append(R"glsl(u; offset > 0u; offset >>= 1u) {
      const uint tree = (lane + 1u) * offset * 2u - 1u;
      if (tree < )glsl") ||
      !backend_source_recipe::append_decimal(sink, shape.width()) ||
      !sink.append(R"glsl(u) {
        const )glsl") ||
      !sink.append(scalar) ||
      !sink.append(R"glsl( prior = range_scan[tree - offset];
        range_scan[tree - offset] = range_scan[tree];
        range_scan[tree] += prior;
      }
      barrier();
    }
    if (is_active) { scratch0_values[uint(index)] = range_scan[lane] + value; }
    return;
  }
  if (params.stage == )glsl") ||
      !backend_source_recipe::append_decimal(
          sink, VulkanRangeStageValue(RangeStageKind::PrefixFixup)) ||
      !sink.append(R"glsl(u) {
    const uint64_t index = group_base + uint64_t(lane);
    if (index < params.stage_element_count && gl_WorkGroupID.x != 0u) {
      scratch0_values[uint(index)] += scratch1_values[gl_WorkGroupID.x - 1u];
    }
    return;
  }
  const uint64_t index = group_base + uint64_t(lane);
  if (index >= params.output_count) { return; }
  const uint64_t anchor = index * params.stride;
  const uint64_t left = anchor < params.padding ? uint64_t(0)
                                                 : anchor - params.padding;
  const uint64_t right_width = params.window_size - params.padding;
  const uint64_t right =
      anchor >= params.input_count
          ? params.input_count - uint64_t(1)
          : (right_width >= params.input_count - anchor
                 ? params.input_count - uint64_t(1)
                 : anchor + right_width - uint64_t(1));
  )glsl") ||
      !sink.append(scalar) ||
      !sink.append(R"glsl( value = scratch0_values[uint(right)];
  if (left != uint64_t(0)) { value -= scratch0_values[uint(left - uint64_t(1))]; }
  )glsl")) {
    return false;
  }
  if (boundary == RangeBoundary::Clamp &&
      (!sink.append(R"glsl(  const uint64_t left_missing =
      anchor < params.padding ? params.padding - anchor : uint64_t(0);
  const uint64_t right_missing =
      anchor >= params.input_count
          ? anchor - params.input_count + right_width
          : (right_width > params.input_count - anchor
                 ? right_width - (params.input_count - anchor)
                 : uint64_t(0));
  if (left_missing != uint64_t(0)) {
    value += )glsl") ||
       !sink.append(scalar) ||
       !sink.append(R"glsl((left_missing * input_values[0u]);
  }
  if (right_missing != uint64_t(0)) {
    value += )glsl") ||
       !sink.append(scalar) || !sink.append(R"glsl((right_missing *
             input_values[uint(params.input_count - uint64_t(1))]);
  }
)glsl"))) {
    return false;
  }
  if (!sink.append(R"glsl(  output_values[uint(index)] = value;
}
)glsl")) {
    return false;
  }
  return true;
}

} // namespace rund::node::accel::detail
