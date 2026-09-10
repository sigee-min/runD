#include "internal.hpp"

namespace rund::node::accel::detail::device_vsm_graph_pointwise::vulkan {

bool append_chained_value(
    std::string &source, const rund::kernel::ArtifactKey &key,
    const rund::kernel::compute_lowering_detail::ParsedIR &parsed,
    const DeviceVsmGraphPointwiseStageTopology &topology,
    const std::span<const std::string> external_values,
    const std::span<const std::string> stage_values, std::string &mapped) {
  namespace lowering = rund::kernel::compute_lowering_detail;
  const std::string value_type =
      key.scalar == rund::kernel::ComputeScalar::Lane32 ? "uint" : "uint64_t";
  const auto layouts = lowering::BuildBindingLayout(parsed);
  if (layouts.size() != parsed.bindings.size()) {
    return false;
  }
  std::vector<std::string> names(parsed.nodes.size() + 1u);
  std::size_t read_ordinal = 0u;
  for (std::size_t index = 0u; index < parsed.nodes.size(); ++index) {
    const lowering::ParsedNode &node = parsed.nodes[index];
    const auto current = static_cast<std::uint32_t>(index + 1u);
    const auto op = static_cast<rund::kernel::IrOp>(node.op);
    if (op == rund::kernel::IrOp::Read) {
      if (read_ordinal >= topology.input_count ||
          node.aux >= parsed.bindings.size() ||
          parsed.bindings[node.aux].kind != lowering::kReadBindingKind) {
        return false;
      }
      const DeviceVsmGraphValueSource input = topology.inputs[read_ordinal++];
      const std::string *value = nullptr;
      if (input.kind == DeviceVsmGraphValueSourceKind::ExternalInput &&
          input.index < external_values.size() &&
          !external_values[input.index].empty()) {
        value = &external_values[input.index];
      } else if (input.kind == DeviceVsmGraphValueSourceKind::StageOutput &&
                 input.index < stage_values.size() &&
                 !stage_values[input.index].empty()) {
        value = &stage_values[input.index];
      }
      if (value == nullptr) {
        return false;
      }
      lowering::AppendVulkanAssignedValue(
          source, key.scalar, lowering::SetVulkanNodeName(names, current),
          *value);
    } else if (op == rund::kernel::IrOp::Write) {
      if (index + 1u != parsed.nodes.size() || node.lhs >= names.size() ||
          names[node.lhs].empty()) {
        return false;
      }
      mapped = names[node.lhs];
      return read_ordinal == topology.input_count;
    } else {
      lowering::AppendVulkanNode(source, parsed, key, layouts, node, current,
                                 names);
    }
  }
  return false;
}

} // namespace rund::node::accel::detail::device_vsm_graph_pointwise::vulkan
