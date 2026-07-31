#pragma once

#include <kernel/program/compute/lowering/fixed/ops.hpp>
#include <kernel/program/compute/lowering/vulkan/syntax.hpp>

namespace rund::kernel {
namespace compute_lowering_detail {

inline void AppendVulkanLoadFromData(std::string &out,
                                     const ComputeScalar scalar,
                                     const std::string &data_symbol) {
  out += "  const uint word = byte_offset >> 2u;\n";
  if (scalar == ComputeScalar::Lane64) {
    out += "  const uint64_t packed = uint64_t(";
    out += data_symbol;
    out += "[word]) | (uint64_t(";
    out += data_symbol;
    out += "[word + 1u]) << 32ul);\n";
    out += "  return packed;\n";
  } else {
    out += "  return ";
    out += data_symbol;
    out += "[word];\n";
  }
}

inline void AppendVulkanParamLoadBody(std::string &out,
                                      const ComputeScalar scalar) {
  out += VulkanType(scalar);
  out += " ";
  out += VulkanParamLoadFunction(scalar);
  out += "(uint byte_offset) {\n";
  AppendVulkanLoadFromData(out, scalar, "rund_params_data");
  out += "}\n";
}

inline void AppendVulkanReadLoadBody(std::string &out,
                                     const ComputeScalar scalar,
                                     const BindingLayout &layout) {
  out += VulkanType(scalar);
  out += " ";
  out += VulkanLoadFunctionName(scalar, layout);
  out += "(uint byte_offset) {\n";
  AppendVulkanLoadFromData(out, scalar, VulkanDataSymbol(layout));
  out += "}\n";
}

inline void AppendVulkanStoreBody(std::string &out, const ComputeScalar scalar,
                                  const BindingLayout &layout) {
  const std::string data_symbol = VulkanDataSymbol(layout);
  out += "void ";
  out += VulkanStoreFunctionName(scalar, layout);
  out += "(uint byte_offset, ";
  out += VulkanType(scalar);
  out += " value) {\n";
  out += "  const uint word = byte_offset >> 2u;\n";
  if (scalar == ComputeScalar::Lane64) {
    out += "  const uint64_t packed = uint64_t(value);\n";
    out += "  ";
    out += data_symbol;
    out += "[word] = uint(packed & 0xfffffffful);\n";
    out += "  ";
    out += data_symbol;
    out += "[word + 1u] = uint((packed >> 32ul) & 0xfffffffful);\n";
  } else {
    out += "  ";
    out += data_symbol;
    out += "[word] = uint(value);\n";
  }
  out += "}\n";
}

inline void AppendVulkanBuffers(std::string &out, const ParsedIR &parsed,
                                const std::vector<BindingLayout> &layouts) {
  out += "layout(set = 0, binding = 0, std430) readonly buffer RundParams {\n";
  out += "  uint rund_params_data[];\n";
  out += "};\n";
  for (std::size_t index = 0u; index < parsed.bindings.size(); ++index) {
    const ParsedBinding &binding = parsed.bindings[index];
    if (binding.kind != 2u) {
      continue;
    }
    const BindingLayout &layout = layouts[index];
    out += "layout(set = 0, binding = ";
    out += std::to_string(layout.buffer);
    out += ", std430) readonly buffer ";
    out += layout.symbol;
    out += "_Buffer {\n";
    out += "  uint ";
    out += VulkanDataSymbol(layout);
    out += "[];\n";
    out += "};\n";
  }
  for (std::size_t index = 0u; index < parsed.bindings.size(); ++index) {
    const ParsedBinding &binding = parsed.bindings[index];
    if (binding.kind != 3u) {
      continue;
    }
    const BindingLayout &layout = layouts[index];
    out += "layout(set = 0, binding = ";
    out += std::to_string(layout.buffer);
    out += ", std430) buffer ";
    out += layout.symbol;
    out += "_Buffer {\n";
    out += "  uint ";
    out += VulkanDataSymbol(layout);
    out += "[];\n";
    out += "};\n";
  }
}

inline void AppendVulkanHelpers(std::string &out, const ParsedIR &parsed,
                                const ArtifactKey &key,
                                const std::vector<BindingLayout> &layouts) {
  AppendVulkanParamLoadBody(out, key.scalar);
  for (std::size_t index = 0u; index < parsed.bindings.size(); ++index) {
    const ParsedBinding &binding = parsed.bindings[index];
    if (binding.kind == 2u) {
      AppendVulkanReadLoadBody(
          out, binding.element_bytes == sizeof(u64) ? ComputeScalar::Lane64
                                                    : ComputeScalar::Lane32,
          layouts[index]);
    }
  }
  for (std::size_t index = 0u; index < parsed.bindings.size(); ++index) {
    const ParsedBinding &binding = parsed.bindings[index];
    if (binding.kind == 3u) {
      AppendVulkanStoreBody(out,
                            VulkanStoreScalar(binding.element_bytes),
                            layouts[index]);
    }
  }
  AppendVulkanFixedOpHelpers(out, parsed, key);
}

} // namespace compute_lowering_detail

} // namespace rund::kernel
