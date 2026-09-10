#pragma once

#include "../../../kernel/backend/source/sink.hpp"
#include "algebra.hpp"

namespace rund::node::accel::detail {

template <typename Sink>
[[nodiscard]] bool EmitVulkanRangeDirectBody(
    Sink &sink, const RangeOp op, const bool wide, const bool signed_values,
    const bool saturating, const RangeBoundary boundary,
    const RangeExec
        &shape) noexcept(noexcept(sink.append(std::string_view{}))) {
  const char *const scalar = VulkanRangeScalar(wide, signed_values);
  if (!sink.append(R"glsl(void main() {
  const uint lane = gl_LocalInvocationID.x;
  const uint64_t group_base = uint64_t(gl_WorkGroupID.x) * uint64_t()glsl") ||
      !backend_source_recipe::append_decimal(sink, shape.width()) ||
      !sink.append(R"glsl();
  const uint gid = uint(group_base + uint64_t(lane));
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
      !EmitVulkanRangeUpdate(sink, op, saturating) || !sink.append(R"glsl(    }
  }
  output_values[gid] = value;
}
)glsl")) {
    return false;
  }
  return true;
}

} // namespace rund::node::accel::detail
