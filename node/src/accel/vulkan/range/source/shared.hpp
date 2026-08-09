#pragma once

#include "../../../kernel/backend/source_recipe.hpp"
#include "algebra.hpp"

namespace rund::node::accel::detail {

template <typename Sink>
[[nodiscard]] bool EmitVulkanRangeSharedBody(
    Sink &sink, const RangeOp op, const bool wide, const bool signed_values,
    const bool saturating,
    const RangeExec
        &shape) noexcept(noexcept(sink.append(std::string_view{}))) {
  const char *const scalar = VulkanRangeScalar(wide, signed_values);
  if (!sink.append(R"glsl(void main() {
  const uint lane = gl_LocalInvocationID.x;
  const uint64_t group_base = uint64_t(gl_WorkGroupID.x) * uint64_t()glsl") ||
      !backend_source_recipe::append_decimal(sink, shape.width()) ||
      !sink.append(R"glsl();
  const uint64_t active_lanes =
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
      !EmitVulkanRangeUpdate(sink, op, saturating) || !sink.append(R"glsl(    }
    output_values[gid] = value;
}
)glsl")) {
    return false;
  }
  return true;
}

} // namespace rund::node::accel::detail
