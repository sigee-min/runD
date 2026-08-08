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
[[nodiscard]] bool EmitVulkanStencilSharedBody(
    Sink &sink, const rund::kernel::StencilOp op, const bool wide,
    const bool signed_extrema,
    const rund::node::accel::detail::StencilGpuShape
        shape) noexcept(noexcept(sink.append(std::string_view{}))) {
  using namespace rund::node::accel::detail;
  if (!sink.append(R"glsl(  const uint64_t active_lanes =
      group_base >= params.element_count
          ? uint64_t(0)
          : min(params.element_count - group_base, uint64_t()glsl") ||
      !backend_source_recipe::append_decimal(sink, shape.width()) ||
      !sink.append(R"glsl());
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
      !backend_source_recipe::append_decimal(sink, shape.radius_cap()) ||
      !sink.append(R"glsl(u + lane] = center_value;
      if (left_inputs == 0u && lane == 0u) {
        for (uint slot = 0u; uint64_t(slot) < params.radius; ++slot) {
          stencil_tile[)glsl") ||
      !backend_source_recipe::append_decimal(sink, shape.radius_cap()) ||
      !sink.append(R"glsl(u - uint(params.radius) + slot] = center_value;
        }
      }
      if (right_inputs == 0u && uint64_t(lane) + uint64_t(1) == active_lanes) {
        for (uint slot = 0u; uint64_t(slot) < params.radius; ++slot) {
          stencil_tile[)glsl") ||
      !backend_source_recipe::append_decimal(sink, shape.radius_cap()) ||
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
      !backend_source_recipe::append_decimal(sink, shape.radius_cap()) ||
      !sink.append(R"glsl(u - left_inputs + lane] = left_value;
      if (lane == 0u && uint64_t(left_inputs) < params.radius) {
        for (uint slot = 0u;
             uint64_t(slot) < params.radius - uint64_t(left_inputs); ++slot) {
          stencil_tile[)glsl") ||
      !backend_source_recipe::append_decimal(sink, shape.radius_cap()) ||
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
      !backend_source_recipe::append_decimal(sink, shape.radius_cap()) ||
      !sink.append(R"glsl(u + uint(active_lanes) + lane] = right_value;
      if (lane + 1u == right_inputs && uint64_t(right_inputs) < params.radius) {
        for (uint slot = right_inputs; uint64_t(slot) < params.radius; ++slot) {
          stencil_tile[)glsl") ||
      !backend_source_recipe::append_decimal(sink, shape.radius_cap()) ||
      !sink.append(R"glsl(u + uint(active_lanes) + slot] = right_value;
        }
      }
    }
    barrier();
    if (uint64_t(lane) >= active_lanes) { return; }
    const uint gid = uint(group_base + uint64_t(lane));
    const uint center = )glsl") ||
      !backend_source_recipe::append_decimal(sink, shape.radius_cap()) ||
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
}
)glsl")) {
    return false;
  }
  return true;
}

template <typename Sink>
[[nodiscard]] bool EmitVulkanStencilDirectBody(
    Sink &sink, const rund::kernel::StencilOp op, const bool wide,
    const bool
        signed_extrema) noexcept(noexcept(sink.append(std::string_view{}))) {
  if (!sink.append(R"glsl(  const uint gid = uint(group_base + uint64_t(lane));
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

[[nodiscard]] constexpr std::uint32_t VulkanStencilStageValue(
    const rund::node::accel::detail::RangeAggregateStageDisposition
        stage) noexcept {
  return static_cast<std::uint32_t>(stage);
}

template <typename Sink>
[[nodiscard]] bool EmitVulkanPrefixDifferenceBody(
    Sink &sink, const bool wide,
    const rund::node::accel::detail::StencilGpuShape
        shape) noexcept(noexcept(sink.append(std::string_view{}))) {
  using namespace rund::node::accel::detail;
  const char *const scalar = wide ? "uint64_t" : "uint";
  if (!sink.append(R"glsl(void main() {
  const uint lane = gl_LocalInvocationID.x;
  const uint64_t group_base = uint64_t(gl_WorkGroupID.x) * uint64_t()glsl") ||
      !backend_source_recipe::append_decimal(sink, shape.width()) ||
      !sink.append(R"glsl();
  if (params.stage == )glsl") ||
      !backend_source_recipe::append_decimal(
          sink, VulkanStencilStageValue(
                    RangeAggregateStageDisposition::PrefixBlock)) ||
      !sink.append(R"glsl(u || params.stage == )glsl") ||
      !backend_source_recipe::append_decimal(
          sink, VulkanStencilStageValue(
                    RangeAggregateStageDisposition::PrefixSummary)) ||
      !sink.append(R"glsl(u) {
    const uint64_t index = group_base + uint64_t(lane);
    const bool is_active = index < params.stage_element_count;
    const )glsl") ||
      !sink.append(scalar) || !sink.append(R"glsl( value = is_active
        ? (params.stage == )glsl") ||
      !backend_source_recipe::append_decimal(
          sink, VulkanStencilStageValue(
                    RangeAggregateStageDisposition::PrefixBlock)) ||
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
          sink, VulkanStencilStageValue(
                    RangeAggregateStageDisposition::PrefixFixup)) ||
      !sink.append(R"glsl(u) {
    const uint64_t index = group_base + uint64_t(lane);
    if (index < params.stage_element_count && gl_WorkGroupID.x != 0u) {
      scratch0_values[uint(index)] += scratch1_values[gl_WorkGroupID.x - 1u];
    }
    return;
  }
  const uint64_t index = group_base + uint64_t(lane);
  if (index >= params.element_count) { return; }
  const uint64_t left = index < params.radius ? uint64_t(0)
                                                : index - params.radius;
  const uint64_t right = min(params.element_count - uint64_t(1),
                             index + params.radius);
  )glsl") ||
      !sink.append(scalar) ||
      !sink.append(R"glsl( value = scratch0_values[uint(right)];
  if (left != uint64_t(0)) { value -= scratch0_values[uint(left - uint64_t(1))]; }
  if (index < params.radius) {
    value += )glsl") ||
      !sink.append(scalar) ||
      !sink.append(R"glsl((params.radius - index) * input_values[0u];
  }
  if (index + params.radius >= params.element_count) {
    value += )glsl") ||
      !sink.append(scalar) ||
      !sink.append(
          R"glsl((index + params.radius - (params.element_count - uint64_t(1))) *
             input_values[uint(params.element_count - uint64_t(1))];
  }
  output_values[uint(index)] = value;
}
)glsl")) {
    return false;
  }
  return true;
}

template <typename Sink>
[[nodiscard]] bool EmitVulkanBlockPrefixSuffixBody(
    Sink &sink, const rund::kernel::StencilOp op,
    const rund::node::accel::detail::StencilGpuShape
        shape) noexcept(noexcept(sink.append(std::string_view{}))) {
  using namespace rund::node::accel::detail;
  const char *const combine =
      op == rund::kernel::StencilOp::Min ? "min" : "max";
  if (!sink.append(R"glsl(void main() {
  const uint64_t block = uint64_t(gl_WorkGroupID.x) * uint64_t()glsl") ||
      !backend_source_recipe::append_decimal(sink, shape.width()) ||
      !sink.append(R"glsl() + uint64_t(gl_LocalInvocationID.x);
  if (params.stage == )glsl") ||
      !backend_source_recipe::append_decimal(
          sink, VulkanStencilStageValue(
                    RangeAggregateStageDisposition::BlockPrefixSuffix)) ||
      !sink.append(R"glsl(u) {
    if (block >= params.stage_aux_count) { return; }
    const uint64_t window = params.radius * uint64_t(2) + uint64_t(1);
    const uint64_t begin = block * window;
    const uint64_t end = min(begin + window, params.stage_element_count);
    for (uint64_t index = begin; index < end; ++index) {
      const value_type value = index < params.radius
          ? input_values[0u]
          : (index - params.radius < params.element_count
                 ? input_values[uint(index - params.radius)]
                 : input_values[uint(params.element_count - uint64_t(1))]);
      if (index == begin) { scratch0_values[uint(index)] = value; }
      else { scratch0_values[uint(index)] = )glsl") ||
      !sink.append(combine) ||
      !sink.append(R"glsl((scratch0_values[uint(index - uint64_t(1))], value); }
    }
    for (uint64_t cursor = end; cursor > begin;) {
      const uint64_t index = cursor - uint64_t(1);
      const value_type value = index < params.radius
          ? input_values[0u]
          : (index - params.radius < params.element_count
                 ? input_values[uint(index - params.radius)]
                 : input_values[uint(params.element_count - uint64_t(1))]);
      if (index + uint64_t(1) == end) { scratch1_values[uint(index)] = value; }
      else { scratch1_values[uint(index)] = )glsl") ||
      !sink.append(combine) ||
      !sink.append(R"glsl((value, scratch1_values[uint(index + uint64_t(1))]); }
      cursor = index;
    }
    return;
  }
  if (block >= params.element_count) { return; }
  const uint64_t right = block + params.radius * uint64_t(2);
  output_values[uint(block)] = )glsl") ||
      !sink.append(combine) ||
      !sink.append(
          R"glsl((scratch1_values[uint(block)], scratch0_values[uint(right)]);
}
)glsl")) {
    return false;
  }
  return true;
}

template <typename Sink>
[[nodiscard]] bool EmitVulkanStencilBody(
    Sink &sink, const rund::kernel::StencilOp op, const bool wide,
    const bool signed_extrema,
    const rund::node::accel::detail::StencilGpuShape
        shape) noexcept(noexcept(sink.append(std::string_view{}))) {
  using namespace rund::node::accel::detail;
  if (!shape.valid() || !sink.append(R"glsl(void main() {
  const uint lane = gl_LocalInvocationID.x;
  const uint64_t group_base = uint64_t(gl_WorkGroupID.x) * uint64_t()glsl") ||
      !backend_source_recipe::append_decimal(sink, shape.width()) ||
      !sink.append(");\n")) {
    return false;
  }
  if (shape.uses_shared_memory()) {
    return EmitVulkanStencilSharedBody(sink, op, wide, signed_extrema, shape);
  }
  return EmitVulkanStencilDirectBody(sink, op, wide, signed_extrema);
}
