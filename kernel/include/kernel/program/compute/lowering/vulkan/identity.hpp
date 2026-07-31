#pragma once

#include <kernel/program/compute/lowering/format.hpp>
#include <kernel/program/compute/lowering/vulkan/syntax.hpp>

namespace rund::kernel {
namespace compute_lowering_detail {

[[nodiscard]] inline std::string
VulkanIdentity(const ParsedIR &parsed, const ArtifactKey &key,
               const std::vector<BindingLayout> &layouts) {
  std::string out;
  out += "rund.compute.vulkan.identity\n";
  out += "kind=vulkan_pipeline_identity\n";
  out += "not_executable_spirv=true\n";
  out += "operation_hex=";
  out += HexText(parsed.name);
  out += "\n";
  AppendKey(out, key, "");
  AppendBindingLayout(out, parsed, layouts, "");
  AppendNodeLayout(out, parsed, "");
  return out;
}

} // namespace compute_lowering_detail

} // namespace rund::kernel
