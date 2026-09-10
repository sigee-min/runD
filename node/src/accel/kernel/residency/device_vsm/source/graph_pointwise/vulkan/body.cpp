#include "internal.hpp"

namespace rund::node::accel::detail::device_vsm_graph_pointwise::vulkan {

bool append_body(
    std::string &source, const rund::kernel::ArtifactKey &key,
    const std::span<const DeviceVsmGraphPointwiseStage> stages,
    const DeviceVsmGraphPointwiseTopology &topology,
    const DeviceVsmPageMap &page_map,
    const DeviceVsmGraphWavefrontProof &wavefront,
    const rund::kernel::compute_lowering_detail::ParsedIR &io,
    const std::vector<rund::kernel::compute_lowering_detail::BindingLayout>
        &layouts,
    const std::span<const std::string> external_values,
    const MapRows &map_rows) {
  namespace lowering = rund::kernel::compute_lowering_detail;
  namespace graph_wavefront = device_vsm_graph_map_reduce;
  const std::string value_type =
      key.scalar == rund::kernel::ComputeScalar::Lane32 ? "uint" : "uint64_t";
  const bool map_active = device_vsm_page_map_active(page_map);
  graph_wavefront::append_vulkan_wavefront_declarations(source);
  device_vsm_typed_map::append_vulkan_main(source);
  source += "  // rund.compute.device_vsm.graph_pointwise.fused_dag\n";
  graph_wavefront::append_vulkan_wavefront_state(source);
  source += "  for (uint batch = 0u; batch < " +
            std::to_string(wavefront.batch_count) + "u; ++batch) {\n";
  graph_wavefront::append_vulkan_wavefront_batch_begin(source, wavefront);
  source += "        if (selected == " + std::to_string(wavefront.map_stage) +
            "u) {\n"
            "          const uint first_page = batch * " +
            std::to_string(wavefront.frame_capacity) +
            "u;\n"
            "          const uint page_limit = min(config.page_count, "
            "first_page + " +
            std::to_string(wavefront.frame_capacity) +
            "u);\n"
            "          for (uint page = first_page; page < page_limit; "
            "++page) {\n"
            "            const uint page_begin = page * "
            "config.payload_elements;\n"
            "            const uint remaining = config.logical_elements - "
            "page_begin;\n"
            "            const uint page_elements = min(remaining, "
            "config.payload_elements);\n"
            "            const uint slot = page % config.width;\n"
            "            const uint turn = page / config.width;\n"
            "            if (local == 0u) {\n"
            "              const uint prior_turn = "
            "atomicAdd(ring_state[slot], 1u);\n"
            "              if (prior_turn != turn) "
            "atomicAdd(result[6], 1u);\n"
            "            }\n"
            "            memoryBarrierBuffer(); barrier();\n"
            "            const uint payload_words = "
            "config.payload_elements * config.element_words;\n"
            "            for (uint page_local = local; page_local < "
            "page_elements; page_local += 256u) {\n"
            "              const uint global_word = "
            "(page_begin + page_local) * config.element_words;\n"
            "              const uint scratch_word = slot * payload_words + "
            "page_local * config.element_words;\n"
            "              for (uint word = 0u; word < config.element_words; "
            "++word) {\n";
  for (std::size_t input = 0u; input < topology.external_input_count; ++input) {
    if (!map_active || map_rows[input] == DeviceVsmPageMapRowCapacity) {
      source += "                input_scratch_" + std::to_string(input) +
                "[scratch_word + word] = input_alias_" + std::to_string(input) +
                "[global_word + word];\n";
      continue;
    }
    DeviceVsmPageMapRow row{};
    if (!device_vsm_page_map_load_row(page_map, map_rows[input], row)) {
      return false;
    }
    const std::string suffix = std::to_string(input);
    const std::string base = "map_base_" + suffix;
    const std::string target = "map_target_" + suffix;
    const std::string count = "map_count_" + suffix;
    const std::string source_page = "map_source_" + suffix;
    const std::string found = "map_found_" + suffix;
    source += "            const uint " + base + " = 8u + " +
              std::to_string(map_rows[input]) +
              "u * 102u;\n"
              "            const uint " +
              target + " = page % " + std::to_string(row.frame_capacity) +
              "u;\n"
              "            const uint map_batch_" +
              suffix + " = (page / " + std::to_string(row.frame_capacity) +
              "u) * " + std::to_string(row.frame_capacity) +
              "u;\n"
              "            const uint " +
              count + " = map_batch_" + suffix +
              " < config.page_count ? min(config.page_count - map_batch_" +
              suffix + ", " + std::to_string(row.frame_capacity) +
              "u) : 0u;\n"
              "            uint " +
              source_page + " = " + target + "; bool " + found +
              " = false;\n"
              "            for (uint map_entry_" +
              suffix + " = 0u; map_entry_" + suffix + " < " +
              std::to_string(row.count) + "u; ++map_entry_" + suffix +
              ") {\n"
              "              const uint map_at_" +
              suffix + " = " + base + " + 6u + map_entry_" + suffix +
              " * 3u;\n"
              "              if (graph_page_map[map_at_" +
              suffix + "] == " + target +
              ") {\n"
              "                const uint entry_source_" +
              suffix + " = graph_page_map[map_at_" + suffix +
              " + 1u];\n"
              "                const uint map_origin_" +
              suffix + " = graph_page_map[map_at_" + suffix +
              " + 2u];\n"
              "                if (entry_source_" +
              suffix + " < " + count + " && map_origin_" + suffix +
              " == 0u) { " + source_page + " = entry_source_" + suffix + "; " +
              found +
              " = true; }\n"
              "                else if (entry_source_" +
              suffix + " < " + count + " && map_origin_" + suffix +
              " == 1u) { " + source_page + " = " + count +
              " - 1u - entry_source_" + suffix + "; " + found +
              " = true; }\n"
              "              }\n"
              "            }\n"
              "            if (!" +
              found + " || " + target + " >= " + count +
              ") atomicAdd(result[6], 1u);\n"
              "            const uint map_global_" +
              suffix + " = (map_batch_" + suffix + " + " + source_page +
              ") * config.payload_elements + page_local;\n"
              "            input_scratch_" +
              suffix + "[scratch_word + word] = input_alias_" + suffix +
              "[map_global_" + suffix + " * config.element_words + word];\n";
  }
  source += "              }\n"
            "            }\n"
            "            memoryBarrierBuffer(); barrier();\n"
            "            for (uint page_local = local; page_local < "
            "page_elements; page_local += 256u) {\n";
  if (map_active) {
    source += "              const uint scratch_word = slot * "
              "config.payload_elements * config.element_words + page_local * "
              "config.element_words;\n";
  }
  source += "              const uint gid = ";
  source += map_active ? "page * config.payload_elements + page_local;\n"
                       : "slot * config.payload_elements + page_local;\n";
  if (map_active) {
    source += "              const uint store_gid = slot * "
              "config.payload_elements + page_local;\n";
  }
  source += "              {\n";
  std::vector<std::string> stage_values(stages.size());
  for (std::size_t stage = 0u; stage < stages.size(); ++stage) {
    if (stages[stage].artifact == nullptr || stages[stage].input == nullptr) {
      return false;
    }
    stage_values[stage] = "rund_graph_stage_" + std::to_string(stage);
    source += "                " + value_type + " " + stage_values[stage] +
              ";\n"
              "                {\n";
    std::string mapped;
    if (!append_chained_value(source, stages[stage].artifact->key,
                              stages[stage].input->parsed,
                              topology.stages[stage], external_values,
                              {stage_values.data(), stage}, mapped)) {
      return false;
    }
    source += "                  " + stage_values[stage] + " = " + mapped +
              ";\n"
              "                }\n";
  }
  device_vsm_typed_map::append_vulkan_store(source, io, layouts,
                                            map_active ? "store_gid" : "gid",
                                            stage_values.back(), key.scalar);
  source += "              }\n"
            "            }\n"
            "            memoryBarrierBuffer(); barrier();\n"
            "            for (uint page_local = local; page_local < "
            "page_elements; page_local += 256u) {\n"
            "              const uint global_word = "
            "(page_begin + page_local) * config.element_words;\n"
            "              const uint scratch_word = slot * payload_words + "
            "page_local * config.element_words;\n"
            "              for (uint word = 0u; word < config.element_words; "
            "++word) output_alias[global_word + word] = "
            "ring_scratch[scratch_word + word];\n"
            "            }\n"
            "            memoryBarrierBuffer(); barrier();\n"
            "            if (local == 0u) { for (uint word = 0u; word < 6u; "
            "++word) atomicAdd(result[word], 1u);\n"
            "              if (page >= config.width) "
            "atomicAdd(result[10], 1u);\n"
            "              const uint schedule = 3u * (page + 1u) + "
            "5u * (slot + 1u) + 7u * (turn + 1u) + "
            "11u * page_elements;\n"
            "              atomicAdd(result[11], schedule);\n"
            "              atomicAdd(result[12], ";
  source += std::to_string(topology.external_input_count + 1u);
  source += "u); }\n"
            "            memoryBarrierBuffer(); barrier();\n"
            "          }\n"
            "        }\n";
  graph_wavefront::append_vulkan_wavefront_batch_end(source);
  source += "  }\n";
  graph_wavefront::append_vulkan_wavefront_final(source);
  source += "}\n";
  return true;
}

} // namespace rund::node::accel::detail::device_vsm_graph_pointwise::vulkan
