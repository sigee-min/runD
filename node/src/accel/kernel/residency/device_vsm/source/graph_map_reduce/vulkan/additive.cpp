#include "../internal.hpp"

#include "../wavefront.hpp"

namespace rund::node::accel::detail::device_vsm_graph_map_reduce {

std::string vulkan_additive_source(
    const rund::kernel::ArtifactKey &key,
    const rund::kernel::compute_lowering_detail::ParsedIR &parsed,
    const rund::kernel::ReduceOp operation,
    const DeviceVsmGraphWavefrontProof &wavefront) {
  const bool sum = operation == rund::kernel::ReduceOp::Sum;
  if (!sum && operation != rund::kernel::ReduceOp::CountNonzero) {
    return {};
  }
  std::string source;
  std::vector<rund::kernel::compute_lowering_detail::BindingLayout> layouts;
  if (!device_vsm_typed_map::append_vulkan_prelude(source, key, parsed,
                                                   layouts)) {
    return {};
  }
  append_vulkan_wavefront_declarations(source);
  device_vsm_typed_map::append_vulkan_main(source);
  source +=
      sum ? "  // rund.compute.device_vsm.graph_map_reduce.sum\n"
          : "  // rund.compute.device_vsm.graph_map_reduce.count_nonzero\n";
  append_vulkan_wavefront_state(source);
  source += "  uint64_t run_low = 0ul; uint64_t run_high = 0ul;\n"
            "  for (uint batch = 0u; batch < ";
  source += std::to_string(wavefront.batch_count);
  source += "u; ++batch) {\n"
            "    uint64_t batch_low = 0ul; uint64_t batch_high = 0ul;\n";
  append_vulkan_wavefront_batch_begin(source, wavefront);
  source += "        if (selected == ";
  source += std::to_string(wavefront.map_stage);
  source += "u) {\n"
            "          const uint first_page = batch * ";
  source += std::to_string(wavefront.frame_capacity);
  source += "u;\n"
            "          const uint page_limit = min(config.page_count, "
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
  source += "              const uint64_t value = ";
  source += sum ? mapped + ";\n" : "uint64_t(" + mapped + " != 0ul);\n";
  source += "              const uint64_t next = batch_low + value;\n"
            "              batch_high += uint64_t(next < batch_low); "
            "batch_low = next;\n"
            "            }\n"
            "            memoryBarrierBuffer(); barrier();\n"
            "            if (local == 0u) { for (uint word = 0u; word < 4u; "
            "++word) atomicAdd(result[word], 1u); }\n"
            "            memoryBarrierBuffer(); barrier();\n"
            "          }\n"
            "        } else if (selected == ";
  source += std::to_string(wavefront.collective_stage);
  source +=
      "u) {\n"
      "          partial_low[local] = batch_low; "
      "partial_high[local] = batch_high; barrier();\n"
      "          for (uint stride = 128u; stride != 0u; stride >>= 1u) {\n"
      "            if (local < stride) {\n"
      "              const uint64_t left = partial_low[local];\n"
      "              const uint64_t combined = left + partial_low[local + "
      "stride];\n"
      "              partial_high[local] += partial_high[local + stride] + "
      "uint64_t(combined < left);\n"
      "              partial_low[local] = combined;\n"
      "            }\n"
      "            barrier();\n"
      "          }\n"
      "          if (local == 0u) {\n"
      "            const uint64_t combined = run_low + partial_low[0];\n"
      "            run_high += partial_high[0] + "
      "uint64_t(combined < run_low);\n"
      "            run_low = combined;\n"
      "          }\n"
      "        }\n";
  append_vulkan_wavefront_batch_end(source);
  source += "  }\n";
  append_vulkan_wavefront_final(source);
  source += "  if (local == 0u) {\n";
  if (sum) {
    source += "    if (run_high != 0ul) { atomicExchange(result[6], 1u); "
              "return; }\n";
  }
  device_vsm_typed_map::append_vulkan_store(source, parsed, layouts, "0u",
                                            "run_low");
  source += "    atomicAdd(result[4], 1u); atomicAdd(result[5], 1u);\n"
            "  }\n"
            "}\n";
  return source;
}

} // namespace rund::node::accel::detail::device_vsm_graph_map_reduce
