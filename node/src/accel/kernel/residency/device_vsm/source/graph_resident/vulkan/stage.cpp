#include "../internal.hpp"

#include <kernel/program/compute/lowering/vulkan/source.hpp>

#include <vector>

namespace rund::node::accel::detail::device_vsm_graph_resident_source::detail {

bool append_stage(std::string &source,
                  const DeviceVsmGraphResidentProof &proof,
                  const DeviceVsmGraphResidentStage &stage,
                  const DeviceVsmGraphResidentStageSource &input,
                  const rund::kernel::ArtifactKey &key,
                  const std::string &element, const std::string &bank,
                  const std::uint32_t frame_capacity,
                  const std::string &slot) {
  namespace lowering = rund::kernel::compute_lowering_detail;
  if (input.artifact == nullptr || input.input == nullptr ||
      input.semantic == nullptr) {
    return false;
  }
  const std::vector<lowering::BindingLayout> no_layouts{};
  std::vector<std::string> names(input.input->parsed.nodes.size() + 1u);
  std::size_t read = 0u;
  const auto port_for =
      [&](const std::uint8_t access,
          const std::size_t ordinal) -> const DeviceVsmGraphResidentPort * {
    for (std::size_t index = 0u; index < stage.port_count; ++index) {
      const auto &candidate = stage.ports[index];
      if (candidate.valid != 0u && candidate.access == access &&
          candidate.program_port == ordinal) {
        return &candidate;
      }
    }
    return nullptr;
  };
  for (std::size_t index = 0u; index < input.input->parsed.nodes.size();
       ++index) {
    const lowering::ParsedNode &node = input.input->parsed.nodes[index];
    const std::uint32_t current = static_cast<std::uint32_t>(index + 1u);
    const auto op = static_cast<rund::kernel::IrOp>(node.op);
    if (op == rund::kernel::IrOp::Read) {
      const auto *const port = port_for(0u, read);
      if (port == nullptr || node.aux >= input.input->parsed.bindings.size() ||
          input.input->parsed.bindings[node.aux].kind !=
              lowering::kReadBindingKind) {
        return false;
      }
      const std::string name = "graph_value_" + std::to_string(current);
      if (!append_load(source, proof, port->resource, element, bank,
                       frame_capacity, slot, name)) {
        return false;
      }
      names[current] = name;
      ++read;
    } else if (op == rund::kernel::IrOp::Write) {
      if (node.lhs >= names.size() || names[node.lhs].empty()) {
        return false;
      }
      const auto *const output = port_for(1u, 0u);
      if (output == nullptr ||
          !append_store(source, proof, output->resource, element, bank,
                        frame_capacity, slot, names[node.lhs])) {
        return false;
      }
      return read + 1u == stage.port_count;
    } else {
      lowering::AppendVulkanNode(source, input.input->parsed, key, no_layouts,
                                 node, current, names);
    }
  }
  return false;
}

} // namespace rund::node::accel::detail::device_vsm_graph_resident_source::detail
