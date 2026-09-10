#include "../internal.hpp"

namespace rund::node::accel::detail::device_vsm_graph_resident_source::detail {

bool append_metal_store(std::string &source,
                        const DeviceVsmGraphResidentProof &proof,
                        const std::uint32_t id, const std::string &element,
                        const std::string &bank,
                        const std::uint32_t frame_capacity,
                        const std::string &slot, const std::string &value) {
  const auto *const row = resource(proof, id);
  if (row == nullptr) {
    return false;
  }
  const std::string value_type = graph_resident_metal_type(proof);
  if (row->role == 2u) {
    if (row->external_slot == DeviceVsmGraphResidentExternalSlotInvalid) {
      return false;
    }
    source += "                graph_endpoint_" +
              std::to_string(row->external_slot) +
              "[gid] = as_type<" + value_type + ">(" + value + ");\n";
    return true;
  }
  if (row->role != 1u || row->owner_slot >= proof.owner_binding_count) {
    return false;
  }
  const auto &owner = proof.owners[row->owner_slot];
  source +=
      "                if (graph_table.valid == 0u || "
      "graph_table.owner_count <= " +
      std::to_string(row->owner_slot) + "u || graph_table.owner_valid[" +
      std::to_string(row->owner_slot) +
      "] == 0u || graph_table.owner_generations[" +
      std::to_string(row->owner_slot * DeviceVsmGraphResidentBankCapacity) +
      "u + " + bank +
      "] == 0ul) atomic_store_explicit(&result[6], 1u, "
      "memory_order_relaxed);\n"
      "                else if (" +
      bank + " == 0u) graph_owner_" + std::to_string(row->owner_slot) + "_0[" +
      offset(owner.bank_regions[0u], owner.cache_regions[0u], frame_capacity,
             slot, element) +
      "] = as_type<" + value_type + ">(" + value +
      ");\n"
      "                else graph_owner_" +
      std::to_string(row->owner_slot) + "_1[" +
      offset(owner.bank_regions[1u], owner.cache_regions[1u], frame_capacity,
             slot, element) +
      "] = as_type<" + value_type + ">(" + value + ");\n";
  return true;
}

bool append_metal_mapped_load(
    std::string &source, const DeviceVsmGraphResidentProof &proof,
    const DeviceVsmGraphResidentResource &row,
    const std::uint32_t frame_capacity, const std::string &name) {
  DeviceVsmPageMapRow map_row{};
  std::size_t map_row_index = DeviceVsmPageMapRowCapacity;
  for (std::size_t index = 0u; index < DeviceVsmPageMapRowCapacity; ++index) {
    DeviceVsmPageMapRow candidate{};
    if (device_vsm_page_map_load_row(proof.page_map, index, candidate) &&
        candidate.resource == row.resource &&
        candidate.external_slot == row.external_slot) {
      map_row = candidate;
      map_row_index = index;
      break;
    }
  }
  if (map_row_index == DeviceVsmPageMapRowCapacity) {
    source += "                const " +
              std::string{graph_resident_metal_type(proof)} + " " + name +
              " = as_type<" + graph_resident_metal_type(proof) +
              ">(graph_endpoint_" +
              std::to_string(row.external_slot) + "[gid]);\n";
    return true;
  }
  if (map_row.frame_capacity == 0u ||
      map_row.frame_capacity != frame_capacity) {
    return false;
  }
  const std::string prefix = name + "_map_";
  const std::string row_base = prefix + "row";
  const std::string page = prefix + "page";
  const std::string local = prefix + "local";
  const std::string batch = prefix + "batch";
  const std::string base = prefix + "base";
  const std::string count = prefix + "count";
  const std::string target = prefix + "target";
  const std::string source_page = prefix + "source";
  const std::string found = prefix + "found";
  const std::string ok = prefix + "ok";
  const std::string entry = prefix + "entry";
  const std::string gid = prefix + "gid";
  source +=
      "                " + std::string{graph_resident_metal_type(proof)} +
      " " + name + " = " + graph_resident_zero_literal(proof) + ";\n" +
      "                const uint " + row_base + " = 8u + " +
      std::to_string(map_row_index) + "u * 102u;\n" +
      "                const uint " + page +
      " = gid / graph_table.payload_elements;\n" +
      "                const uint " + local +
      " = gid % graph_table.payload_elements;\n" +
      "                const uint " + batch + " = " + page + " / " +
      std::to_string(map_row.frame_capacity) + "u;\n" +
      "                const uint " + base + " = " + batch + " * " +
      std::to_string(map_row.frame_capacity) + "u;\n" +
      "                const uint " + count + " = " + base +
      " < graph_table.page_count ? min(graph_table.page_count - " + base +
      ", " + std::to_string(map_row.frame_capacity) + "u) : 0u;\n" +
      "                const uint " + target + " = " + page + " % " +
      std::to_string(map_row.frame_capacity) + "u;\n" +
      "                uint " + source_page + " = " + target + ";\n" +
      "                bool " + found + " = false;\n" +
      "                bool " + ok +
      " = graph_page_map[0u] == 0x504d4150u && "
      "graph_page_map[1u] == 1u && graph_page_map[2u] == 1u && "
      "graph_page_map[" +
      row_base + "] == " + std::to_string(map_row.resource) +
      "u && graph_page_map[" + row_base +
      " + 1u] == " + std::to_string(map_row.external_slot) +
      "u && graph_page_map[" + row_base +
      " + 2u] == " + std::to_string(map_row.frame_capacity) +
      "u && graph_page_map[" + row_base + " + 3u] == " +
      std::to_string(static_cast<std::uint32_t>(map_row.page_bytes)) +
      "u && graph_page_map[" + row_base + " + 4u] == " +
      std::to_string(static_cast<std::uint32_t>(map_row.page_bytes >> 32u)) +
      "u && graph_page_map[" + row_base +
      " + 5u] == " + std::to_string(map_row.count) + "u;\n" +
      "                for (uint " + entry + " = 0u; " + entry + " < " +
      std::to_string(map_row.frame_capacity) + "u; ++" + entry + ") {\n" +
      "                  const uint " + prefix + "at = " + row_base +
      " + 6u + " + entry + " * 3u;\n" +
      "                  if (graph_page_map[" + prefix + "at] == " + target +
      ") {\n" + "                    const uint " + prefix +
      "source = graph_page_map[" + prefix + "at + 1u];\n" +
      "                    const uint " + prefix + "origin = graph_page_map[" +
      prefix + "at + 2u];\n" + "                    if (" + prefix +
      "origin == 0u && " + prefix + "source < " + count + ") { " + source_page +
      " = " + prefix + "source; " + found + " = true; }\n" +
      "                    else if (" + prefix + "origin == 1u && " + count +
      " != 0u && " + prefix + "source < " + count + ") { " + source_page +
      " = " + count + " - 1u - " + prefix + "source; " + found +
      " = true; }\n" +
      "                  }\n"
      "                }\n" +
      "                " + ok + " = " + ok + " && " + count + " != 0u && " +
      target + " < " + count + " && " + found + " && " + source_page + " < " +
      count + ";\n" + "                if (!" + ok +
      ") "
      "atomic_store_explicit(&result[6], 1u, memory_order_relaxed);\n" +
      "                else { const uint " + gid + " = (" + base + " + " +
      source_page + ") * graph_table.payload_elements + " + local + "; " +
      name + " = as_type<" + graph_resident_metal_type(proof) +
      ">(graph_endpoint_" +
      std::to_string(row.external_slot) + "[" + gid + "]); }\n";
  return true;
}

} // namespace rund::node::accel::detail::device_vsm_graph_resident_source::detail
