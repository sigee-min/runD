#pragma once

inline void AppendVulkanStencilBody(
    std::string &source, const rund::kernel::StencilOp op, const bool u64,
    const bool signed_extrema) {
  source += "void main() {\n";
  source += "  const uint gid = gl_GlobalInvocationID.x;\n";
  source += "  if (uint64_t(gid) >= params.element_count) { return; }\n";
  if (signed_extrema) {
    source += u64 ? "  int64_t value = int64_t(input_values[gid]);\n"
                  : "  int value = int(input_values[gid]);\n";
  } else {
    source += "  uint64_t value = uint64_t(input_values[gid]);\n";
  }
  source +=
      "  for (uint64_t step = uint64_t(1); step <= params.radius; ++step) {\n";
  source += "    const uint left = uint64_t(gid) < step ? 0u : "
            "uint(uint64_t(gid) - step);\n";
  source += "    const uint right = uint64_t(gid) + step >= "
            "params.element_count ? uint(params.element_count - uint64_t(1)) : "
            "uint(uint64_t(gid) + step);\n";
  if (op == rund::kernel::StencilOp::Min) {
    source +=
        signed_extrema
            ? (u64 ? "    value = min(value, min(int64_t(input_values[left]), "
                     "int64_t(input_values[right])));\n"
                   : "    value = min(value, min(int(input_values[left]), "
                     "int(input_values[right])));\n")
            : "    value = min(value, min(uint64_t(input_values[left]), "
              "uint64_t(input_values[right])));\n";
  } else if (op == rund::kernel::StencilOp::Max) {
    source +=
        signed_extrema
            ? (u64 ? "    value = max(value, max(int64_t(input_values[left]), "
                     "int64_t(input_values[right])));\n"
                   : "    value = max(value, max(int(input_values[left]), "
                     "int(input_values[right])));\n")
            : "    value = max(value, max(uint64_t(input_values[left]), "
              "uint64_t(input_values[right])));\n";
  } else {
    source += "    value += uint64_t(input_values[left]) + "
              "uint64_t(input_values[right]);\n";
  }
  source += "  }\n";
  source += "  output_values[gid] = ";
  source += u64 ? "uint64_t(value)" : "uint(value)";
  source += ";\n";
  source += "}\n";
}
