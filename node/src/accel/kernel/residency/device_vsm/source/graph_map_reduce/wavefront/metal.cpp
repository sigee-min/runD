#include "../wavefront.hpp"

#include "model.hpp"

namespace rund::node::accel::detail::device_vsm_graph_map_reduce {

void append_metal_wavefront_state(std::string &source) {
  source +=
      "  threadgroup uint wavefront_order[8];\n"
      "  threadgroup uint wavefront_previous;\n"
      "  threadgroup uint wavefront_steps;\n"
      "  threadgroup uint wavefront_trace;\n"
      "  threadgroup uint wavefront_valid;\n"
      "  if (local == 0u) { wavefront_previous = 0u; wavefront_steps = 0u; "
      "wavefront_trace = 2166136261u; wavefront_valid = 1u; }\n"
      "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
}

void append_metal_wavefront_batch_begin(
    std::string &source, const DeviceVsmGraphWavefrontProof &wavefront) {
  const std::uint32_t all = (std::uint32_t{1u} << wavefront.stage_count) - 1u;
  source += "    if (local == 0u) {\n"
            "      const uint previous = wavefront_previous;\n"
            "      uint completed = 0u; bool map_selected = false;\n"
            "      bool collective_selected = false;\n"
            "      for (uint ordinal = 0u; ordinal < ";
  source += std::to_string(wavefront.stage_count);
  source += "u; ++ordinal) {\n        uint selected = ";
  source += std::to_string(wavefront.stage_count);
  source += "u;\n";
  append_wavefront_candidates(source, wavefront, "&&");
  source += "        wavefront_order[ordinal] = selected;\n"
            "        if (selected == ";
  source += std::to_string(wavefront.stage_count);
  source += "u) { wavefront_valid = 0u; } else {\n"
            "          if (selected == ";
  source += std::to_string(wavefront.collective_stage);
  source += "u && !map_selected) wavefront_valid = 0u;\n"
            "          map_selected = map_selected || selected == ";
  source += std::to_string(wavefront.map_stage);
  source += "u; collective_selected = collective_selected || selected == ";
  source += std::to_string(wavefront.collective_stage);
  source += "u;\n"
            "          completed |= 1u << selected; ++wavefront_steps;\n"
            "          const uint coordinate = batch * ";
  source += std::to_string(wavefront.stage_count);
  source +=
      "u + selected + 1u;\n"
      "          wavefront_trace = (wavefront_trace ^ coordinate) * "
      "16777619u;\n"
      "        }\n      }\n      wavefront_valid = (wavefront_valid != 0u && "
      "completed == ";
  source += std::to_string(all);
  source += "u && map_selected && collective_selected) ? 1u : 0u;\n"
            "      wavefront_previous = completed;\n    }\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    if (wavefront_valid != 0u) {\n"
            "      for (uint ordinal = 0u; ordinal < ";
  source += std::to_string(wavefront.stage_count);
  source += "u; ++ordinal) {\n"
            "        const uint selected = wavefront_order[ordinal];\n";
}

void append_metal_wavefront_batch_end(std::string &source) {
  source += "        threadgroup_barrier(mem_flags::mem_device | "
            "mem_flags::mem_threadgroup);\n"
            "      }\n    }\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
}

void append_metal_wavefront_final(std::string &source) {
  source += "  if (local == 0u) {\n"
            "    atomic_store_explicit(&result[8], wavefront_valid != 0u ? "
            "wavefront_steps : 0u, memory_order_relaxed);\n"
            "    atomic_store_explicit(&result[9], wavefront_valid != 0u ? "
            "wavefront_trace : 0u, memory_order_relaxed);\n"
            "  }\n"
            "  threadgroup_barrier(mem_flags::mem_device | "
            "mem_flags::mem_threadgroup);\n"
            "  if (wavefront_valid == 0u) return;\n";
}

} // namespace rund::node::accel::detail::device_vsm_graph_map_reduce
