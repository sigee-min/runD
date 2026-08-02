#pragma once

#include <string_view>

template <typename Sink>
[[nodiscard]] bool EmitVulkanStencilBody(
    Sink &sink, const rund::kernel::StencilOp op, const bool wide,
    const bool signed_extrema)
    noexcept(noexcept(sink.append(std::string_view{}))) {
  if (!sink.append(R"glsl(void main() {
  const uint gid = gl_GlobalInvocationID.x;
  if (uint64_t(gid) >= params.element_count) { return; }
)glsl") ||
      !sink.append(signed_extrema
                       ? (wide
                              ? "  int64_t value = int64_t(input_values[gid]);\n"
                              : "  int value = int(input_values[gid]);\n")
                       : "  uint64_t value = uint64_t(input_values[gid]);\n") ||
      !sink.append(R"glsl(  for (uint64_t step = uint64_t(1); step <= params.radius; ++step) {
    const uint left = uint64_t(gid) < step ? 0u : uint(uint64_t(gid) - step);
    const uint right = uint64_t(gid) + step >= params.element_count ? uint(params.element_count - uint64_t(1)) : uint(uint64_t(gid) + step);
)glsl")) {
    return false;
  }
  if (op == rund::kernel::StencilOp::Min) {
    if (!sink.append(
            signed_extrema
                ? (wide
                       ? "    value = min(value, min(int64_t(input_values[left]), int64_t(input_values[right])));\n"
                       : "    value = min(value, min(int(input_values[left]), int(input_values[right])));\n")
                : "    value = min(value, min(uint64_t(input_values[left]), uint64_t(input_values[right])));\n")) {
      return false;
    }
  } else if (op == rund::kernel::StencilOp::Max) {
    if (!sink.append(
            signed_extrema
                ? (wide
                       ? "    value = max(value, max(int64_t(input_values[left]), int64_t(input_values[right])));\n"
                       : "    value = max(value, max(int(input_values[left]), int(input_values[right])));\n")
                : "    value = max(value, max(uint64_t(input_values[left]), uint64_t(input_values[right])));\n")) {
      return false;
    }
  } else if (!sink.append(
                 "    value += uint64_t(input_values[left]) + uint64_t(input_values[right]);\n")) {
    return false;
  }
  return sink.append("  }\n  output_values[gid] = ") &&
         sink.append(wide ? "uint64_t(value)" : "uint(value)") &&
         sink.append(";\n}\n");
}
