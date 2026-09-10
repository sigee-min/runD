#include "../internal.hpp"

#include "../wavefront.hpp"

namespace rund::node::accel::detail::device_vsm_graph_map_reduce {

std::string metal_additive_source(
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
  if (!device_vsm_typed_map::append_metal_header(source, key, parsed,
                                                 layouts)) {
    return {};
  }
  source +=
      sum ? "  // rund.compute.device_vsm.graph_map_reduce.sum\n"
          : "  // rund.compute.device_vsm.graph_map_reduce.count_nonzero\n";
  append_metal_wavefront_state(source);
  source += "  threadgroup ulong partial_low[256];\n"
            "  threadgroup ulong partial_high[256];\n"
            "  ulong run_low = 0ul; ulong run_high = 0ul;\n"
            "  for (uint batch = 0u; batch < ";
  source += std::to_string(wavefront.batch_count);
  source += "u; ++batch) {\n"
            "    ulong batch_low = 0ul; ulong batch_high = 0ul;\n";
  append_metal_wavefront_batch_begin(source, wavefront);
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
            "            const ulong page_begin = ulong(page) * "
            "ulong(config.payload_elements);\n"
            "            const ulong remaining = config.logical_elements - "
            "page_begin;\n"
            "            const uint page_elements = uint(min(remaining, "
            "ulong(config.payload_elements)));\n"
            "            for (uint page_local = local; page_local < "
            "page_elements; page_local += 256u) {\n"
            "              const ulong gid = page_begin + "
            "ulong(page_local);\n";
  std::string mapped;
  if (!device_vsm_typed_map::append_metal_value(source, key, parsed, layouts,
                                                mapped)) {
    return {};
  }
  source += "              const ulong value = ";
  source += sum ? mapped + ";\n" : "ulong(" + mapped + " != 0ul);\n";
  source += "              const ulong next = batch_low + value;\n"
            "              batch_high += ulong(next < batch_low); "
            "batch_low = next;\n"
            "            }\n"
            "            threadgroup_barrier(mem_flags::mem_device | "
            "mem_flags::mem_threadgroup);\n"
            "            if (local == 0u) { for (uint word = 0u; word < 4u; "
            "++word) atomic_fetch_add_explicit(&result[word], 1u, "
            "memory_order_relaxed); }\n"
            "            threadgroup_barrier(mem_flags::mem_device | "
            "mem_flags::mem_threadgroup);\n"
            "          }\n"
            "        } else if (selected == ";
  source += std::to_string(wavefront.collective_stage);
  source +=
      "u) {\n"
      "          partial_low[local] = batch_low; "
      "partial_high[local] = batch_high;\n"
      "          threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "          for (uint stride = 128u; stride != 0u; stride >>= 1u) {\n"
      "            if (local < stride) {\n"
      "              const ulong left = partial_low[local];\n"
      "              const ulong combined = left + partial_low[local + "
      "stride];\n"
      "              partial_high[local] += partial_high[local + stride] + "
      "ulong(combined < left);\n"
      "              partial_low[local] = combined;\n"
      "            }\n"
      "            threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "          }\n"
      "          if (local == 0u) {\n"
      "            const ulong combined = run_low + partial_low[0];\n"
      "            run_high += partial_high[0] + ulong(combined < run_low);\n"
      "            run_low = combined;\n"
      "          }\n"
      "        }\n";
  append_metal_wavefront_batch_end(source);
  source += "  }\n";
  append_metal_wavefront_final(source);
  source += "  if (local == 0u) {\n";
  if (sum) {
    source += "    if (run_high != 0ul) { "
              "atomic_store_explicit(&result[6], 1u, memory_order_relaxed); "
              "return; }\n";
  }
  device_vsm_typed_map::append_metal_store(source, parsed, layouts, "0ul",
                                           "run_low");
  source += "    atomic_fetch_add_explicit(&result[4], 1u, "
            "memory_order_relaxed);\n"
            "    atomic_fetch_add_explicit(&result[5], 1u, "
            "memory_order_relaxed);\n"
            "  }\n"
            "}\n";
  return source;
}

} // namespace rund::node::accel::detail::device_vsm_graph_map_reduce
