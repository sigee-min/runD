#include "../internal.hpp"

#include "../wavefront.hpp"

namespace rund::node::accel::detail::device_vsm_graph_map_reduce {

std::string vulkan_extreme_source(
    const rund::kernel::ArtifactKey &key,
    const rund::kernel::compute_lowering_detail::ParsedIR &parsed,
    const rund::kernel::ReduceOp operation,
    const DeviceVsmGraphWavefrontProof &wavefront) {
  const bool minimum = operation == rund::kernel::ReduceOp::Min;
  if (!minimum && operation != rund::kernel::ReduceOp::Max) {
    return {};
  }
  const char *const identity = minimum ? "0xfffffffffffffffful" : "0ul";
  const char *const combine = minimum ? "min" : "max";
  std::string source;
  std::vector<rund::kernel::compute_lowering_detail::BindingLayout> layouts;
  if (!device_vsm_typed_map::append_vulkan_prelude(source, key, parsed,
                                                   layouts)) {
    return {};
  }
  append_vulkan_wavefront_declarations(source);
  device_vsm_typed_map::append_vulkan_main(source);
  source += minimum ? "  // rund.compute.device_vsm.graph_map_reduce.min\n"
                    : "  // rund.compute.device_vsm.graph_map_reduce.max\n";
  append_vulkan_wavefront_state(source);
  source += "  uint64_t value = ";
  source += identity;
  source += ";\n  for (uint batch = 0u; batch < ";
  source += std::to_string(wavefront.batch_count);
  source += "u; ++batch) {\n    uint64_t batch_value = ";
  source += identity;
  source += ";\n";
  append_vulkan_wavefront_batch_begin(source, wavefront);
  source += "        if (selected == ";
  source += std::to_string(wavefront.map_stage);
  source += "u) {\n          const uint first_page = batch * ";
  source += std::to_string(wavefront.frame_capacity);
  source += "u;\n          const uint page_limit = min(config.page_count, "
            "first_page + ";
  source += std::to_string(wavefront.frame_capacity);
  source += "u);\n"
            "          for (uint page = first_page; page < page_limit; "
            "++page) {\n"
            "            const uint page_begin = page * "
            "config.payload_elements;\n"
            "            const uint remaining = config.logical_elements - "
            "page_begin;\n"
            "            const uint page_elements = min(remaining, "
            "config.payload_elements);\n"
            "            for (uint page_local = local; page_local < "
            "page_elements; page_local += 256u) {\n"
            "              const uint gid = page_begin + page_local;\n";
  std::string mapped;
  if (!device_vsm_typed_map::append_vulkan_value(source, key, parsed, layouts,
                                                 mapped)) {
    return {};
  }
  source += "              batch_value = ";
  source += combine;
  source += "(batch_value, " + mapped +
            ");\n"
            "            }\n"
            "            memoryBarrierBuffer(); barrier();\n"
            "            if (local == 0u) { for (uint word = 0u; word < "
            "4u; ++word) atomicAdd(result[word], 1u); }\n"
            "            memoryBarrierBuffer(); barrier();\n"
            "          }\n"
            "        } else if (selected == ";
  source += std::to_string(wavefront.collective_stage);
  source += "u) {\n"
            "          partial[local] = batch_value; barrier();\n"
            "          for (uint stride = 128u; stride != 0u; stride >>= "
            "1u) {\n"
            "            if (local < stride) { partial[local] = ";
  source += combine;
  source += "(partial[local], partial[local + stride]); }\n"
            "            barrier();\n"
            "          }\n"
            "          if (local == 0u) value = ";
  source += combine;
  source += "(value, partial[0]);\n        }\n";
  append_vulkan_wavefront_batch_end(source);
  source += "  }\n";
  append_vulkan_wavefront_final(source);
  source += "  if (local == 0u) {\n";
  device_vsm_typed_map::append_vulkan_store(source, parsed, layouts, "0u",
                                            "value");
  source += "    atomicAdd(result[4], 1u); atomicAdd(result[5], 1u);\n"
            "  }\n"
            "}\n";
  return source;
}

} // namespace rund::node::accel::detail::device_vsm_graph_map_reduce
