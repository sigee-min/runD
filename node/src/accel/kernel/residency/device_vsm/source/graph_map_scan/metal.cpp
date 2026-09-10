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
  if (!device_vsm_typed_map::append_metal_value(source, key, parsed, layouts,
                                                mapped)) {
    return false;
  }
  source += std::string{"      "} + destination + " = " + mapped + ";\n";
  return true;
}

} // namespace

std::string
metal_source(const rund::kernel::ArtifactKey &key,
             const rund::kernel::compute_lowering_detail::ParsedIR &parsed,
             const rund::kernel::ScanOp operation) {
  if (operation != rund::kernel::ScanOp::InclusiveSum &&
      operation != rund::kernel::ScanOp::ExclusiveSum) {
    return {};
  }
  std::string source;
  std::vector<rund::kernel::compute_lowering_detail::BindingLayout> layouts;
  if (!device_vsm_typed_map::append_metal_header(source, key, parsed,
                                                 layouts)) {
    return {};
  }
  source +=
      "  // rund.compute.device_vsm.graph_map_scan\n"
      "  threadgroup ulong partial_low[256];\n"
      "  threadgroup ulong partial_high[256];\n"
      "  threadgroup ulong carry_low;\n"
      "  threadgroup ulong carry_high;\n"
      "  threadgroup uint overflow;\n"
      "  threadgroup uint failed_page;\n"
      "  ulong lane_low = 0ul; ulong lane_high = 0ul;\n"
      "  for (ulong index = ulong(local); index < config.logical_elements; "
      "index += 256ul) {\n"
      "    const ulong gid = index;\n"
      "    ulong mapped = 0ul;\n";
  if (!append_mapped_value(source, key, parsed, layouts, "mapped")) {
    return {};
  }
  source +=
      "    const ulong next = lane_low + mapped;\n"
      "    lane_high += ulong(next < lane_low); lane_low = next;\n"
      "  }\n"
      "  partial_low[local] = lane_low; partial_high[local] = lane_high;\n"
      "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "  for (uint stride = 128u; stride != 0u; stride >>= 1u) {\n"
      "    if (local < stride) {\n"
      "      const ulong left = partial_low[local];\n"
      "      const ulong combined = left + partial_low[local + stride];\n"
      "      partial_high[local] += partial_high[local + stride] + "
      "ulong(combined < left);\n"
      "      partial_low[local] = combined;\n"
      "    }\n"
      "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "  }\n"
      "  if (local == 0u) { overflow = uint(partial_high[0] != 0ul); }\n"
      "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "  if (local == 0u && overflow != 0u) {\n"
      "    ulong prefix = 0ul; failed_page = 0u;\n"
      "    for (ulong index = 0ul; index < config.logical_elements; ++index) "
      "{\n"
      "      const ulong gid = index;\n"
      "      ulong mapped = 0ul;\n";
  if (!append_mapped_value(source, key, parsed, layouts, "mapped")) {
    return {};
  }
  source +=
      "      const ulong next = prefix + mapped;\n"
      "      if (next < prefix) { failed_page = uint(index / "
      "ulong(config.payload_elements)); break; }\n"
      "      prefix = next;\n"
      "    }\n"
      "  }\n"
      "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "  if (local == 0u) { carry_low = 0ul; carry_high = 0ul; }\n"
      "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "  for (ulong chunk = 0ul; chunk < config.logical_elements; chunk += "
      "256ul) {\n"
      "    const ulong index = chunk + ulong(local);\n"
      "    ulong mapped = 0ul;\n"
      "    if (index < config.logical_elements) {\n"
      "      const ulong gid = index;\n";
  if (!append_mapped_value(source, key, parsed, layouts, "mapped")) {
    return {};
  }
  source +=
      "    }\n"
      "    partial_low[local] = mapped; partial_high[local] = 0ul;\n"
      "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "    for (uint offset = 1u; offset < 256u; offset <<= 1u) {\n"
      "      ulong next_low = partial_low[local];\n"
      "      ulong next_high = partial_high[local];\n"
      "      if (local >= offset) {\n"
      "        const ulong left = partial_low[local - offset];\n"
      "        const ulong combined = left + next_low;\n"
      "        next_high += partial_high[local - offset] + "
      "ulong(combined < left);\n"
      "        next_low = combined;\n"
      "      }\n"
      "      threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "      partial_low[local] = next_low; partial_high[local] = next_high;\n"
      "      threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "    }\n"
      "    if (index < config.logical_elements && overflow == 0u) {\n";
  source += operation == rund::kernel::ScanOp::InclusiveSum
                ? "      const ulong value = partial_low[local] + "
                  "carry_low;\n"
                : "      const ulong value = local == 0u ? carry_low : "
                  "partial_low[local - 1u] + carry_low;\n";
  device_vsm_typed_map::append_metal_store(source, parsed, layouts, "index",
                                           "value");
  source +=
      "    }\n"
      "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "    if (local == 0u) {\n"
      "      const uint count = uint(min(256ul, config.logical_elements - "
      "chunk));\n"
      "      const ulong prior = carry_low;\n"
      "      carry_low = prior + partial_low[count - 1u];\n"
      "      carry_high += partial_high[count - 1u] + "
      "ulong(carry_low < prior);\n"
      "    }\n"
      "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "  }\n"
      "  if (local == 0u && overflow != 0u) {\n"
      "    atomic_store_explicit(&result[0], config.page_count, "
      "memory_order_relaxed);\n"
      "    atomic_store_explicit(&result[1], config.page_count, "
      "memory_order_relaxed);\n"
      "    atomic_store_explicit(&result[2], config.page_count, "
      "memory_order_relaxed);\n"
      "    atomic_store_explicit(&result[3], failed_page, "
      "memory_order_relaxed);\n"
      "    atomic_store_explicit(&result[6], 1u, memory_order_relaxed);\n"
      "    atomic_store_explicit(&result[7], failed_page, "
      "memory_order_relaxed);\n"
      "  } else if (local == 0u) {\n"
      "    for (uint word = 0u; word < 6u; ++word) {\n"
      "      atomic_store_explicit(&result[word], config.page_count, "
      "memory_order_relaxed);\n"
      "    }\n"
      "    atomic_store_explicit(&result[6], 0u, memory_order_relaxed);\n"
      "    atomic_store_explicit(&result[7], 0xffffffffu, "
      "memory_order_relaxed);\n"
      "  }\n"
      "}\n";
  return source;
}

} // namespace rund::node::accel::detail::device_vsm_graph_map_scan
