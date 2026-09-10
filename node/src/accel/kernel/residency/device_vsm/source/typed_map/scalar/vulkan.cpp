#include "../scalar.hpp"

#include <kernel/program/compute/lowering/layout.hpp>
#include <kernel/program/compute/lowering/vulkan/source.hpp>

#include <vector>

namespace rund::node::accel::detail::device_vsm_typed_map {

bool append_vulkan_u32_scalar_function(
    std::string &source, const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission &input,
    const char *const name) {
  namespace lowering = rund::kernel::compute_lowering_detail;
  if (name == nullptr || *name == '\0') {
    return false;
  }
  const auto layouts = lowering::BuildBindingLayout(input.parsed);
  if (layouts.size() != input.parsed.bindings.size()) {
    return false;
  }
  source += "uint ";
  source += name;
  source += "(uint input_value, uint gid) {\n";
  std::vector<std::string> names(input.parsed.nodes.size() + 1u);
  for (std::size_t index = 0u; index < input.parsed.nodes.size(); ++index) {
    const lowering::ParsedNode &node = input.parsed.nodes[index];
    const auto op = static_cast<rund::kernel::IrOp>(node.op);
    const auto current = static_cast<std::uint32_t>(index + 1u);
    if (op == rund::kernel::IrOp::Read) {
      source += "  const uint " + lowering::SetVulkanNodeName(names, current) +
                " = input_value;\n";
    } else if (op == rund::kernel::IrOp::Write) {
      if (node.lhs >= names.size() || names[node.lhs].empty()) {
        return false;
      }
      source += "  return " + names[node.lhs] + ";\n}\n";
      return true;
    } else {
      lowering::AppendVulkanNode(source, input.parsed, artifact.key, layouts,
                                 node, current, names);
    }
  }
  return false;
}

} // namespace rund::node::accel::detail::device_vsm_typed_map
