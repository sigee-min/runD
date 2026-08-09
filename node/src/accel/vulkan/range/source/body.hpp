#pragma once

#include "../../../kernel/backend/source_recipe.hpp"
#include <string_view>

template <typename Sink>
[[nodiscard]] bool EmitVulkanRangeUpdate(
    Sink &sink, const rund::node::accel::detail::RangeOp op, const bool wide,
    const bool saturating) noexcept(noexcept(sink.append(std::string_view{}))) {
  (void)wide;
  if (op == rund::node::accel::detail::RangeOp::Minimum) {
    return sink.append("      value = min(value, item);\n");
  }
  if (op == rund::node::accel::detail::RangeOp::Maximum) {
    return sink.append("      value = max(value, item);\n");
  }
  return sink.append(saturating
                         ? "      value = rund_range_add_sat(value, item);\n"
                         : "      value += item;\n");
}

template <typename Sink>
[[nodiscard]] bool EmitVulkanRangeSharedBody(
    Sink &sink, const rund::node::accel::detail::RangeOp op, const bool wide,
    const bool signed_values, const bool saturating,
    const rund::node::accel::detail::RangeExec
        &shape) noexcept(noexcept(sink.append(std::string_view{}))) {
  using namespace rund::node::accel::detail;
  const char *const scalar =
      signed_values ? (wide ? "int64_t" : "int") : (wide ? "uint64_t" : "uint");
  if (!sink.append(R"glsl(  const uint64_t active_lanes =
      group_base >= params.input_count
          ? uint64_t(0)
          : min(params.input_count - group_base, uint64_t()glsl") ||
      !backend_source_recipe::append_decimal(sink, shape.width()) ||
      !sink.append(R"glsl());
    const uint64_t group_end = group_base + active_lanes;
    const uint left_inputs = uint(min(group_base, params.padding));
    const uint right_inputs = group_end >= params.input_count
                                  ? 0u
                                  : uint(min(params.input_count - group_end,
                                             params.padding));
    if (uint64_t(lane) < active_lanes) {
      const )glsl") ||
      !sink.append(scalar) ||
      !sink.append(
          R"glsl( center_value = input_values[uint(group_base + uint64_t(lane))];
      range_tile[)glsl") ||
      !backend_source_recipe::append_decimal(sink,
                                             shape.shared_radius_capacity()) ||
      !sink.append(R"glsl(u + lane] = center_value;
      if (left_inputs == 0u && lane == 0u) {
        for (uint slot = 0u; uint64_t(slot) < params.padding; ++slot) {
          range_tile[)glsl") ||
      !backend_source_recipe::append_decimal(sink,
                                             shape.shared_radius_capacity()) ||
      !sink.append(R"glsl(u - uint(params.padding) + slot] = center_value;
        }
      }
      if (right_inputs == 0u && uint64_t(lane) + uint64_t(1) == active_lanes) {
        for (uint slot = 0u; uint64_t(slot) < params.padding; ++slot) {
          range_tile[)glsl") ||
      !backend_source_recipe::append_decimal(sink,
                                             shape.shared_radius_capacity()) ||
      !sink.append(R"glsl(u + uint(active_lanes) + slot] = center_value;
        }
      }
    }
    if (lane < left_inputs) {
      const )glsl") ||
      !sink.append(scalar) ||
      !sink.append(
          R"glsl( left_value = input_values[uint(group_base - uint64_t(left_inputs) + uint64_t(lane))];
      range_tile[)glsl") ||
      !backend_source_recipe::append_decimal(sink,
                                             shape.shared_radius_capacity()) ||
      !sink.append(R"glsl(u - left_inputs + lane] = left_value;
      if (lane == 0u && uint64_t(left_inputs) < params.padding) {
        for (uint slot = 0u;
             uint64_t(slot) < params.padding - uint64_t(left_inputs); ++slot) {
          range_tile[)glsl") ||
      !backend_source_recipe::append_decimal(sink,
                                             shape.shared_radius_capacity()) ||
      !sink.append(R"glsl(u - uint(params.padding) + slot] = left_value;
        }
      }
    }
    if (lane < right_inputs) {
      const )glsl") ||
      !sink.append(scalar) ||
      !sink.append(
          R"glsl( right_value = input_values[uint(group_end + uint64_t(lane))];
      range_tile[)glsl") ||
      !backend_source_recipe::append_decimal(sink,
                                             shape.shared_radius_capacity()) ||
      !sink.append(R"glsl(u + uint(active_lanes) + lane] = right_value;
      if (lane + 1u == right_inputs && uint64_t(right_inputs) < params.padding) {
        for (uint slot = right_inputs; uint64_t(slot) < params.padding; ++slot) {
          range_tile[)glsl") ||
      !backend_source_recipe::append_decimal(sink,
                                             shape.shared_radius_capacity()) ||
      !sink.append(R"glsl(u + uint(active_lanes) + slot] = right_value;
        }
      }
    }
    barrier();
    if (uint64_t(lane) >= active_lanes) { return; }
    const uint gid = uint(group_base + uint64_t(lane));
    const uint center = )glsl") ||
      !backend_source_recipe::append_decimal(sink,
                                             shape.shared_radius_capacity()) ||
      !sink.append("u + lane;\n") ||
      !sink.append(
          "    const uint first = center - uint(params.padding);\n    ") ||
      !sink.append(scalar) || !sink.append(" value = range_tile[first];\n") ||
      !sink.append(
          R"glsl(    for (uint64_t slot = uint64_t(1); slot < params.window_size; ++slot) {
      const )glsl") ||
      !sink.append(scalar) ||
      !sink.append(R"glsl( item = range_tile[first + uint(slot)];
)glsl") ||
      !EmitVulkanRangeUpdate(sink, op, wide, saturating) ||
      !sink.append(R"glsl(    }
    output_values[gid] = value;
}
)glsl")) {
    return false;
  }
  return true;
}

template <typename Sink>
[[nodiscard]] bool EmitVulkanRangeDirectBody(
    Sink &sink, const rund::node::accel::detail::RangeOp op, const bool wide,
    const bool signed_values, const bool saturating,
    const rund::node::accel::detail::RangeBoundary
        boundary) noexcept(noexcept(sink.append(std::string_view{}))) {
  using namespace rund::node::accel::detail;
  const char *const scalar =
      signed_values ? (wide ? "int64_t" : "int") : (wide ? "uint64_t" : "uint");
  if (!sink.append(R"glsl(  const uint gid = uint(group_base + uint64_t(lane));
  if (uint64_t(gid) >= params.output_count) { return; }
  const uint64_t anchor = uint64_t(gid) * params.stride;
)glsl") ||
      !sink.append("  ") || !sink.append(scalar) || !sink.append(" value = ") ||
      !sink.append(scalar) || !sink.append(R"glsl((0);
  bool seeded = false;
  for (uint64_t slot = uint64_t(0); slot < params.window_size; ++slot) {
    uint64_t input_index = uint64_t(0);
    bool valid = true;
    if (slot < params.padding) {
      const uint64_t delta = params.padding - slot;
      if (anchor < delta) {
)glsl")) {
    return false;
  }
  if (!sink.append(boundary == RangeBoundary::Clamp
                       ? "        input_index = uint64_t(0);\n"
                       : "        valid = false;\n") ||
      !sink.append(R"glsl(      } else {
        input_index = anchor - delta;
)glsl") ||
      !sink.append(boundary == RangeBoundary::Clamp
                       ? R"glsl(        if (input_index >= params.input_count) {
          input_index = params.input_count - uint64_t(1);
        }
)glsl"
                       : R"glsl(        if (input_index >= params.input_count) {
          valid = false;
        }
)glsl") ||
      !sink.append(R"glsl(      }
    } else {
      const uint64_t delta = slot - params.padding;
      if (anchor >= params.input_count ||
          delta >= params.input_count - anchor) {
)glsl") ||
      !sink.append(
          boundary == RangeBoundary::Clamp
              ? "        input_index = params.input_count - uint64_t(1);\n"
              : "        valid = false;\n") ||
      !sink.append(R"glsl(      } else {
        input_index = anchor + delta;
      }
    }
    if (!valid) { continue; }
    const )glsl") ||
      !sink.append(scalar) ||
      !sink.append(R"glsl( item = input_values[uint(input_index)];
    if (!seeded) {
      value = item;
      seeded = true;
    } else {
)glsl") ||
      !EmitVulkanRangeUpdate(sink, op, wide, saturating) ||
      !sink.append(R"glsl(    }
  }
  output_values[gid] = value;
}
)glsl")) {
    return false;
  }
  return true;
}

[[nodiscard]] constexpr std::uint32_t VulkanRangeStageValue(
    const rund::node::accel::detail::RangeStageKind stage) noexcept {
  return static_cast<std::uint32_t>(stage);
}

template <typename Sink>
[[nodiscard]] bool EmitVulkanPrefixDifferenceBody(
    Sink &sink, const bool wide,
    const rund::node::accel::detail::RangeBoundary boundary,
    const rund::node::accel::detail::RangeExec
        &shape) noexcept(noexcept(sink.append(std::string_view{}))) {
  using namespace rund::node::accel::detail;
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

template <typename Sink>
[[nodiscard]] bool EmitVulkanBlockPrefixSuffixBody(
    Sink &sink, const rund::node::accel::detail::RangeOp op,
    const rund::node::accel::detail::RangeBoundary boundary, const bool wide,
    const bool signed_values,
    const rund::node::accel::detail::RangeExec
        &shape) noexcept(noexcept(sink.append(std::string_view{}))) {
  using namespace rund::node::accel::detail;
  const char *const combine =
      op == rund::node::accel::detail::RangeOp::Minimum ? "min" : "max";
  const char *const identity =
      op == RangeOp::Minimum
          ? (signed_values
                 ? (wide ? "int64_t(0x7fffffffffffffffUL)" : "2147483647")
                 : (wide ? "uint64_t(0xffffffffffffffffUL)" : "0xffffffffu"))
          : (signed_values
                 ? (wide ? "(-int64_t(0x7fffffffffffffffUL) - int64_t(1))"
                         : "(-2147483647 - 1)")
                 : (wide ? "uint64_t(0)" : "0u"));
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

template <typename Sink>
[[nodiscard]] bool EmitVulkanRangeBody(
    Sink &sink, const rund::node::accel::detail::RangeOp op, const bool wide,
    const bool signed_values, const bool saturating,
    const rund::node::accel::detail::RangeBoundary boundary,
    const rund::node::accel::detail::RangeExec
        &shape) noexcept(noexcept(sink.append(std::string_view{}))) {
  using namespace rund::node::accel::detail;
  if (!sink.append(R"glsl(void main() {
  const uint lane = gl_LocalInvocationID.x;
  const uint64_t group_base = uint64_t(gl_WorkGroupID.x) * uint64_t()glsl") ||
      !backend_source_recipe::append_decimal(sink, shape.width()) ||
      !sink.append(");\n")) {
    return false;
  }
  if (shape.uses_shared_halo()) {
    return EmitVulkanRangeSharedBody(sink, op, wide, signed_values, saturating,
                                     shape);
  }
  return EmitVulkanRangeDirectBody(sink, op, wide, signed_values, saturating,
                                   boundary);
}
