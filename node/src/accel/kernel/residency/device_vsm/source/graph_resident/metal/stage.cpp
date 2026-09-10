#include "../internal.hpp"

#include <kernel/program/compute/lowering/metal/source.hpp>

#include <vector>

namespace rund::node::accel::detail::device_vsm_graph_resident_source::detail {

bool append_metal_stage(
    std::string &source, const DeviceVsmGraphResidentProof &proof,
    const DeviceVsmGraphResidentStage &stage,
    const DeviceVsmGraphResidentStageSource &input,
    const rund::kernel::ArtifactKey &key, const std::string &element,
    const std::string &slot, const std::string &bank,
    const std::uint32_t frame_capacity) {
  namespace lowering = rund::kernel::compute_lowering_detail;
  if (input.artifact == nullptr || input.input == nullptr ||
      input.semantic == nullptr) {
    return false;
  }
  const auto layouts = lowering::BuildBindingLayout(input.input->parsed);
  if (layouts.size() != input.input->parsed.bindings.size()) {
    return false;
  }
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
    const auto &node = input.input->parsed.nodes[index];
    const std::uint32_t current = static_cast<std::uint32_t>(index + 1u);
    const auto op = static_cast<rund::kernel::IrOp>(node.op);
    if (op == rund::kernel::IrOp::Read) {
      const auto *const port = port_for(0u, read);
      if (port == nullptr || node.aux >= input.input->parsed.bindings.size() ||
          input.input->parsed.bindings[node.aux].kind !=
              lowering::kReadBindingKind) {
        return false;
      }
      const auto *const row = resource(proof, port->resource);
      if (row == nullptr) {
        return false;
      }
      const std::string name = "graph_value_" + std::to_string(current);
      if (row->role == 0u) {
        if (row->external_slot == DeviceVsmGraphResidentExternalSlotInvalid) {
          return false;
        }
        if (device_vsm_page_map_active(proof.page_map)) {
          if (!append_metal_mapped_load(source, proof, *row, frame_capacity,
                                        name)) {
            return false;
          }
        } else {
          source += "                const " +
                    std::string{graph_resident_metal_type(proof)} + " " +
                    name + " = as_type<" + graph_resident_metal_type(proof) +
                    ">(graph_endpoint_" +
                    std::to_string(row->external_slot) + "[gid]);\n";
        }
      } else if (row->role == 1u &&
                 row->owner_slot < proof.owner_binding_count) {
        const auto &owner = proof.owners[row->owner_slot];
        source += "                " +
                  std::string{graph_resident_metal_type(proof)} + " " + name +
                  " = " + graph_resident_zero_literal(proof) + ";\n"
                  "                if (graph_table.valid == 0u || "
                  "graph_table.owner_count <= " +
                  std::to_string(row->owner_slot) +
                  "u || graph_table.owner_valid[" +
                  std::to_string(row->owner_slot) +
                  "] == 0u || graph_table.owner_generations[" +
                  std::to_string(row->owner_slot *
                                 DeviceVsmGraphResidentBankCapacity) +
                  "u + " + bank +
                  "] == 0ul) "
                  "atomic_store_explicit(&result[6], 1u, "
                  "memory_order_relaxed);\n"
                  "                else if (" +
                  bank + " == 0u) " + name + " = as_type<" +
                  graph_resident_metal_type(proof) + ">(graph_owner_" +
                  std::to_string(row->owner_slot) + "_0[(" +
                  std::to_string(owner.bank_regions[0u].first) + "u - " +
                  std::to_string(owner.cache_regions[0u].first) +
                  "u) * graph_table.payload_elements + " + slot +
                  " * graph_table.payload_elements + " + element +
                  "]);\n"
                  "                else " + name + " = as_type<" +
                  graph_resident_metal_type(proof) + ">(graph_owner_" +
                  std::to_string(row->owner_slot) + "_1[(" +
                  std::to_string(owner.bank_regions[1u].first) + "u - " +
                  std::to_string(owner.cache_regions[1u].first) +
                  "u) * graph_table.payload_elements + " + slot +
                  " * graph_table.payload_elements + " + element + "]);\n";
      } else {
        return false;
      }
      names[current] = name;
      ++read;
    } else if (op == rund::kernel::IrOp::Write) {
      if (node.lhs >= names.size() || names[node.lhs].empty()) {
        return false;
      }
      const auto *const output = port_for(1u, 0u);
      if (output == nullptr || read + 1u != stage.port_count ||
          !append_metal_store(source, proof, output->resource, element, bank,
                              frame_capacity, slot, names[node.lhs])) {
        return false;
      }
      return true;
    } else {
      lowering::AppendMetalNode(source, input.input->parsed, key, layouts, node,
                                current, names);
    }
  }
  return false;
}

} // namespace rund::node::accel::detail::device_vsm_graph_resident_source::detail
