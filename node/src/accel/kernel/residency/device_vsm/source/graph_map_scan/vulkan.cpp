#include "internal.hpp"

namespace rund::node::accel::detail::device_vsm_graph_map_scan {
namespace {

[[nodiscard]] bool append_mapped_value(
    std::string &source, const rund::kernel::ArtifactKey &key,
    const rund::kernel::compute_lowering_detail::ParsedIR &parsed,
    const std::vector<rund::kernel::compute_lowering_detail::BindingLayout>
        &layouts,
    const char *const destination) {
  std::string mapped;
  if (!device_vsm_typed_map::append_vulkan_value(source, key, parsed, layouts,
                                                 mapped)) {
    return false;
  }
  source += std::string{"      "} + destination + " = " + mapped + ";\n";
  return true;
}

} // namespace

std::string
vulkan_source(const rund::kernel::ArtifactKey &key,
              const rund::kernel::compute_lowering_detail::ParsedIR &parsed,
              const rund::kernel::ScanOp operation) {
  if (operation != rund::kernel::ScanOp::InclusiveSum &&
      operation != rund::kernel::ScanOp::ExclusiveSum) {
    return {};
  }
  std::string source;
  std::vector<rund::kernel::compute_lowering_detail::BindingLayout> layouts;
  if (!device_vsm_typed_map::append_vulkan_prelude(source, key, parsed,
                                                   layouts)) {
    return {};
  }
  source += "// rund.compute.device_vsm.graph_map_scan\n"
            "shared uint64_t carry_low; shared uint64_t carry_high;\n"
            "shared uint overflow; shared uint failed_page;\n";
  device_vsm_typed_map::append_vulkan_main(source);
  source +=
      "  uint64_t lane_low = 0ul; uint64_t lane_high = 0ul;\n"
      "  for (uint index = local; index < config.logical_elements; index += "
      "256u) {\n"
      "    const uint gid = index;\n"
      "    uint64_t mapped = 0ul;\n";
  if (!append_mapped_value(source, key, parsed, layouts, "mapped")) {
    return {};
  }
  source +=
      "    const uint64_t next = lane_low + mapped;\n"
      "    lane_high += uint64_t(next < lane_low); lane_low = next;\n"
      "  }\n"
      "  partial_low[local] = lane_low; partial_high[local] = lane_high; "
      "barrier();\n"
      "  for (uint stride = 128u; stride != 0u; stride >>= 1u) {\n"
      "    if (local < stride) {\n"
      "      const uint64_t left = partial_low[local];\n"
      "      const uint64_t combined = left + partial_low[local + stride];\n"
      "      partial_high[local] += partial_high[local + stride] + "
      "uint64_t(combined < left);\n"
      "      partial_low[local] = combined;\n"
      "    }\n"
      "    barrier();\n"
      "  }\n"
      "  if (local == 0u) { overflow = uint(partial_high[0] != 0ul); } "
      "barrier();\n"
      "  if (local == 0u && overflow != 0u) {\n"
      "    uint64_t prefix = 0ul; failed_page = 0u;\n"
      "    for (uint index = 0u; index < config.logical_elements; ++index) "
      "{\n"
      "      const uint gid = index;\n"
      "      uint64_t mapped = 0ul;\n";
  if (!append_mapped_value(source, key, parsed, layouts, "mapped")) {
    return {};
  }
  source +=
      "      const uint64_t next = prefix + mapped;\n"
      "      if (next < prefix) { failed_page = index / "
      "config.payload_elements; break; }\n"
      "      prefix = next;\n"
      "    }\n"
      "  }\n"
      "  barrier();\n"
      "  if (local == 0u) { carry_low = 0ul; carry_high = 0ul; } barrier();\n"
      "  for (uint chunk = 0u; chunk < config.logical_elements; chunk += "
      "256u) {\n"
      "    const uint index = chunk + local;\n"
      "    uint64_t mapped = 0ul;\n"
      "    if (index < config.logical_elements) {\n"
      "      const uint gid = index;\n";
  if (!append_mapped_value(source, key, parsed, layouts, "mapped")) {
    return {};
  }
  source +=
      "    }\n"
      "    partial_low[local] = mapped; partial_high[local] = 0ul; barrier();\n"
      "    for (uint offset = 1u; offset < 256u; offset <<= 1u) {\n"
      "      uint64_t next_low = partial_low[local];\n"
      "      uint64_t next_high = partial_high[local];\n"
      "      if (local >= offset) {\n"
      "        const uint64_t left = partial_low[local - offset];\n"
      "        const uint64_t combined = left + next_low;\n"
      "        next_high += partial_high[local - offset] + "
      "uint64_t(combined < left);\n"
      "        next_low = combined;\n"
      "      }\n"
      "      barrier(); partial_low[local] = next_low; "
      "partial_high[local] = next_high; barrier();\n"
      "    }\n"
      "    if (index < config.logical_elements && overflow == 0u) {\n";
  source += operation == rund::kernel::ScanOp::InclusiveSum
                ? "      const uint64_t value = partial_low[local] + "
                  "carry_low;\n"
                : "      const uint64_t value = local == 0u ? carry_low : "
                  "partial_low[local - 1u] + carry_low;\n";
  device_vsm_typed_map::append_vulkan_store(source, parsed, layouts, "index",
                                            "value");
  source +=
      "    }\n"
      "    barrier();\n"
      "    if (local == 0u) {\n"
      "      const uint count = min(256u, config.logical_elements - chunk);\n"
      "      const uint64_t prior = carry_low;\n"
      "      carry_low = prior + partial_low[count - 1u];\n"
      "      carry_high += partial_high[count - 1u] + "
      "uint64_t(carry_low < prior);\n"
      "    }\n"
      "    barrier();\n"
      "  }\n"
      "  if (local == 0u && overflow != 0u) {\n"
      "    atomicExchange(result[0], config.page_count);\n"
      "    atomicExchange(result[1], config.page_count);\n"
      "    atomicExchange(result[2], config.page_count);\n"
      "    atomicExchange(result[3], failed_page);\n"
      "    atomicExchange(result[6], 1u);\n"
      "    atomicExchange(result[7], failed_page);\n"
      "  } else if (local == 0u) {\n"
      "    for (uint word = 0u; word < 6u; ++word) { "
      "atomicExchange(result[word], config.page_count); }\n"
      "    atomicExchange(result[6], 0u);\n"
      "    atomicExchange(result[7], 0xffffffffu);\n"
      "  }\n"
      "}\n";
  return source;
}

} // namespace rund::node::accel::detail::device_vsm_graph_map_scan
