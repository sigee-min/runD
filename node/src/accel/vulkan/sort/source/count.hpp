#pragma once

namespace rund::node::accel::detail {

template <typename Source>
inline void AppendVulkanSortCount(Source &source) {
  source += "layout(set = 0, binding = 8, std430) readonly buffer "
            "LogicalCount { uint logical_count[]; };\n";
  source += "struct SortRange { uint64_t logical; bool invalid; };\n";
  source += "SortRange ResolveRange() {\n";
  source += "  uint64_t count = params.element_count;\n";
  source += "  if (params.count_words == 0u) { "
            "return SortRange(count, false); }\n";
  source += "  count = uint64_t(logical_count[0]);\n";
  source += "  if (params.count_words == 2u) { count |= "
            "uint64_t(logical_count[1]) << 32u; }\n";
  source += "  const bool invalid = count > params.element_count;\n";
  source += "  return SortRange(invalid ? uint64_t(0) : count, invalid);\n}\n";
}

} // namespace rund::node::accel::detail
