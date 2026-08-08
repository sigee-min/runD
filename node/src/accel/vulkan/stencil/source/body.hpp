#pragma once

#include "../../../kernel/backend/source_recipe.hpp"
#include "../../../stencil/shape.hpp"

#include <string_view>

template <typename Sink>
[[nodiscard]] bool EmitVulkanStencilUpdate(
    Sink &sink, const rund::kernel::StencilOp op, const bool wide,
    const bool signed_extrema,
    const bool shared) noexcept(noexcept(sink.append(std::string_view{}))) {
  if (shared) {
    if (op == rund::kernel::StencilOp::Min) {
      return sink.append(
          signed_extrema
              ? (wide ? "      value = min(value, "
                        "min(int64_t(stencil_tile[center - step]), "
                        "int64_t(stencil_tile[center + step])));\n"
                      : "      value = min(value, min(int(stencil_tile[center "
                        "- step]), int(stencil_tile[center + step])));\n")
              : (wide ? "      value = min(value, "
                        "min(uint64_t(stencil_tile[center - step]), "
                        "uint64_t(stencil_tile[center + step])));\n"
                      : "      value = min(value, min(stencil_tile[center - "
                        "step], stencil_tile[center + step]));\n"));
    }
    if (op == rund::kernel::StencilOp::Max) {
      return sink.append(
          signed_extrema
              ? (wide ? "      value = max(value, "
                        "max(int64_t(stencil_tile[center - step]), "
                        "int64_t(stencil_tile[center + step])));\n"
                      : "      value = max(value, max(int(stencil_tile[center "
                        "- step]), int(stencil_tile[center + step])));\n")
              : (wide ? "      value = max(value, "
                        "max(uint64_t(stencil_tile[center - step]), "
                        "uint64_t(stencil_tile[center + step])));\n"
                      : "      value = max(value, max(stencil_tile[center - "
                        "step], stencil_tile[center + step]));\n"));
    }
    return sink.append(
        wide ? "      value += uint64_t(stencil_tile[center - step]) + "
               "uint64_t(stencil_tile[center + step]);\n"
             : "      value += stencil_tile[center - step] + "
               "stencil_tile[center + step];\n");
  }
  if (op == rund::kernel::StencilOp::Min) {
    return sink.append(
        signed_extrema
            ? (wide ? "    value = min(value, min(int64_t(input_values[left]), "
                      "int64_t(input_values[right])));\n"
                    : "    value = min(value, min(int(input_values[left]), "
                      "int(input_values[right])));\n")
            : (wide ? "    value = min(value, "
                      "min(uint64_t(input_values[left]), "
                      "uint64_t(input_values[right])));\n"
                    : "    value = min(value, min(input_values[left], "
                      "input_values[right]));\n"));
  }
  if (op == rund::kernel::StencilOp::Max) {
    return sink.append(
        signed_extrema
            ? (wide ? "    value = max(value, max(int64_t(input_values[left]), "
                      "int64_t(input_values[right])));\n"
                    : "    value = max(value, max(int(input_values[left]), "
                      "int(input_values[right])));\n")
            : (wide ? "    value = max(value, "
                      "max(uint64_t(input_values[left]), "
                      "uint64_t(input_values[right])));\n"
                    : "    value = max(value, max(input_values[left], "
                      "input_values[right]));\n"));
  }
  return sink.append(wide ? "    value += uint64_t(input_values[left]) + "
                            "uint64_t(input_values[right]);\n"
                          : "    value += input_values[left] + "
                            "input_values[right];\n");
}

template <typename Sink>
[[nodiscard]] bool EmitVulkanStencilBody(
    Sink &sink, const rund::kernel::StencilOp op, const bool wide,
    const bool
        signed_extrema) noexcept(noexcept(sink.append(std::string_view{}))) {
  using namespace rund::node::accel::detail;
  if (!sink.append(R"glsl(void main() {
  const uint lane = gl_LocalInvocationID.x;
  const uint64_t group_base = uint64_t(gl_WorkGroupID.x) * uint64_t()glsl") ||
      !backend_source_recipe::append_decimal(sink,
                                             kStencilPhysicalGroupWidth) ||
      !sink.append(R"glsl();
  const uint64_t active_lanes = group_base >= params.element_count
                                    ? uint64_t(0)
                                    : min(params.element_count - group_base,
                                          uint64_t()glsl") ||
      !backend_source_recipe::append_decimal(sink,
                                             kStencilPhysicalGroupWidth) ||
      !sink.append(R"glsl());
  if (params.radius <= uint64_t()glsl") ||
      !backend_source_recipe::append_decimal(sink, kStencilSharedRadiusCap) ||
      !sink.append(R"glsl()) {
    const uint64_t group_end = group_base + active_lanes;
    const uint left_inputs = uint(min(group_base, params.radius));
    const uint right_inputs = group_end >= params.element_count
                                  ? 0u
                                  : uint(min(params.element_count - group_end,
                                             params.radius));
    if (uint64_t(lane) < active_lanes) {
      const )glsl") ||
      !sink.append(wide ? "uint64_t" : "uint") ||
      !sink.append(
          R"glsl( center_value = input_values[uint(group_base + uint64_t(lane))];
      stencil_tile[)glsl") ||
      !backend_source_recipe::append_decimal(sink, kStencilSharedRadiusCap) ||
      !sink.append(R"glsl(u + lane] = center_value;
      if (left_inputs == 0u && lane == 0u) {
        for (uint slot = 0u; uint64_t(slot) < params.radius; ++slot) {
          stencil_tile[)glsl") ||
      !backend_source_recipe::append_decimal(sink, kStencilSharedRadiusCap) ||
      !sink.append(R"glsl(u - uint(params.radius) + slot] = center_value;
        }
      }
      if (right_inputs == 0u && uint64_t(lane) + uint64_t(1) == active_lanes) {
        for (uint slot = 0u; uint64_t(slot) < params.radius; ++slot) {
          stencil_tile[)glsl") ||
      !backend_source_recipe::append_decimal(sink, kStencilSharedRadiusCap) ||
      !sink.append(R"glsl(u + uint(active_lanes) + slot] = center_value;
        }
      }
    }
    if (lane < left_inputs) {
      const )glsl") ||
      !sink.append(wide ? "uint64_t" : "uint") ||
      !sink.append(
          R"glsl( left_value = input_values[uint(group_base - uint64_t(left_inputs) + uint64_t(lane))];
      stencil_tile[)glsl") ||
      !backend_source_recipe::append_decimal(sink, kStencilSharedRadiusCap) ||
      !sink.append(R"glsl(u - left_inputs + lane] = left_value;
      if (lane == 0u && uint64_t(left_inputs) < params.radius) {
        for (uint slot = 0u;
             uint64_t(slot) < params.radius - uint64_t(left_inputs); ++slot) {
          stencil_tile[)glsl") ||
      !backend_source_recipe::append_decimal(sink, kStencilSharedRadiusCap) ||
      !sink.append(R"glsl(u - uint(params.radius) + slot] = left_value;
        }
      }
    }
    if (lane < right_inputs) {
      const )glsl") ||
      !sink.append(wide ? "uint64_t" : "uint") ||
      !sink.append(
          R"glsl( right_value = input_values[uint(group_end + uint64_t(lane))];
      stencil_tile[)glsl") ||
      !backend_source_recipe::append_decimal(sink, kStencilSharedRadiusCap) ||
      !sink.append(R"glsl(u + uint(active_lanes) + lane] = right_value;
      if (lane + 1u == right_inputs && uint64_t(right_inputs) < params.radius) {
        for (uint slot = right_inputs; uint64_t(slot) < params.radius; ++slot) {
          stencil_tile[)glsl") ||
      !backend_source_recipe::append_decimal(sink, kStencilSharedRadiusCap) ||
      !sink.append(R"glsl(u + uint(active_lanes) + slot] = right_value;
        }
      }
    }
    barrier();
    if (uint64_t(lane) >= active_lanes) { return; }
    const uint gid = uint(group_base + uint64_t(lane));
    const uint center = )glsl") ||
      !backend_source_recipe::append_decimal(sink, kStencilSharedRadiusCap) ||
      !sink.append("u + lane;\n") ||
      !sink.append(
          signed_extrema
              ? (wide ? "    int64_t value = int64_t(stencil_tile[center]);\n"
                      : "    int value = int(stencil_tile[center]);\n")
              : (wide ? "    uint64_t value = "
                        "uint64_t(stencil_tile[center]);\n"
                      : "    uint value = stencil_tile[center];\n")) ||
      !sink.append(
          R"glsl(    for (uint step = 1u; uint64_t(step) <= params.radius; ++step) {
)glsl") ||
      !EmitVulkanStencilUpdate(sink, op, wide, signed_extrema, true) ||
      !sink.append("    }\n    output_values[gid] = ") ||
      !sink.append(wide ? "uint64_t(value)" : "uint(value)") ||
      !sink.append(R"glsl(;
    return;
  }
  const uint gid = uint(group_base + uint64_t(lane));
  if (uint64_t(gid) >= params.element_count) { return; }
)glsl") ||
      !sink.append(
          signed_extrema
              ? (wide ? "  int64_t value = int64_t(input_values[gid]);\n"
                      : "  int value = int(input_values[gid]);\n")
              : (wide ? "  uint64_t value = uint64_t(input_values[gid]);\n"
                      : "  uint value = input_values[gid];\n")) ||
      !sink.append(
          R"glsl(  for (uint64_t step = uint64_t(1); step <= params.radius; ++step) {
    const uint left = uint64_t(gid) < step ? 0u : uint(uint64_t(gid) - step);
    const uint right = uint64_t(gid) + step >= params.element_count ? uint(params.element_count - uint64_t(1)) : uint(uint64_t(gid) + step);
)glsl")) {
    return false;
  }
  if (!EmitVulkanStencilUpdate(sink, op, wide, signed_extrema, false)) {
    return false;
  }
  return sink.append("  }\n  output_values[gid] = ") &&
         sink.append(wide ? "uint64_t(value)" : "uint(value)") &&
         sink.append(";\n}\n");
}
