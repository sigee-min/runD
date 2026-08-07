#pragma once

#include "base.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
template <typename Sink>
[[nodiscard]] bool AppendVulkanSortPrefixSource(Sink &sink) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  backend_source_recipe::SourceBuilder source{sink};
  source += "layout(set = 0, binding = 1, std430) buffer BlockCounts {\n";
  source += "  uint block_counts[];\n};\n";
  source += "layout(set = 0, binding = 5, std430) buffer BlockOffsets {\n";
  source += "  uint block_offsets[];\n};\n";
  source += "shared uint local_values[kSortThreadCount];\n";
  source += "shared uint local_carry;\n";
  source += "shared uint local_base;\n";
  source += "void main() {\n";
  source += "  const uint bucket = gl_WorkGroupID.x;\n";
  source += "  const uint tid = gl_LocalInvocationID.x;\n";
  source += "  const SortRange range = ResolveRange();\n";
  source += "  const uint active_blocks = range.logical == uint64_t(0) "
            "? 0u : 1u + (uint(range.logical) - 1u) / kSortBlockSize;\n";
  source += "  if (tid == 0u) { local_carry = 0u; }\n";
  source += "  barrier();\n";
  source += "  for (uint tile = 0u; tile < active_blocks; "
            "tile += kSortThreadCount) {\n";
  source += "    const uint block = tile + tid;\n";
  source += "    const uint entry = bucket * params.block_count + block;\n";
  source += "    const uint value = block < active_blocks "
            "? block_counts[entry] : 0u;\n";
  source += "    local_values[tid] = value;\n";
  source += "    if (tid == 0u) { local_base = local_carry; }\n";
  source += "    barrier();\n";
  source +=
      "    for (uint step = 1u; step < kSortThreadCount; step <<= 1u) {\n";
  source += "      const uint add = tid >= step "
            "? local_values[tid - step] : 0u;\n";
  source += "      barrier();\n";
  source += "      local_values[tid] += add;\n";
  source += "      barrier();\n";
  source += "    }\n";
  source += "    if (block < active_blocks) {\n";
  source += "      block_offsets[entry] = local_base + "
            "local_values[tid] - value;\n";
  source += "    }\n";
  source += "    barrier();\n";
  source += "    if (tid == 0u) {\n";
  source += "      const uint remaining = active_blocks - tile;\n";
  source += "      const uint last = min(remaining, kSortThreadCount) - 1u;\n";
  source += "      local_carry = local_base + local_values[last];\n";
  source += "    }\n";
  source += "    barrier();\n";
  source += "  }\n";
  source += "  if (tid == 0u) {\n";
  source += "    const uint totals = params.block_count * kSortBucketCount;\n";
  source += "    block_counts[totals + bucket] = local_carry;\n";
  source += "  }\n";
  source += "}\n";
  return source.valid();
}

template <typename Sink>
[[nodiscard]] bool AppendVulkanSortBaseOffsetSource(Sink &sink) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  backend_source_recipe::SourceBuilder source{sink};
  source += "layout(set = 0, binding = 1, std430) buffer BlockCounts {\n";
  source += "  uint block_counts[];\n};\n";
  source += "shared uint local_values[kSortBucketCount];\n";
  source += "void main() {\n";
  source += "  const uint tid = gl_LocalInvocationID.x;\n";
  source += "  const uint totals = params.block_count * kSortBucketCount;\n";
  source += "  const uint value = params.block_count == 1u "
            "? block_counts[tid] : block_counts[totals + tid];\n";
  source += "  local_values[tid] = value;\n";
  source += "  barrier();\n";
  source += "  for (uint step = 1u; step < kSortBucketCount; step <<= 1u) {\n";
  source += "    const uint add = tid >= step "
            "? local_values[tid - step] : 0u;\n";
  source += "    barrier();\n";
  source += "    local_values[tid] += add;\n";
  source += "    barrier();\n";
  source += "  }\n";
  source += "  block_counts[totals + tid] = local_values[tid] - value;\n";
  source += "}\n";
  return source.valid();
}
#endif

} // namespace rund::node::accel::detail
