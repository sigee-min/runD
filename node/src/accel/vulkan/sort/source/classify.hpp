#pragma once

#include "base.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
template <typename Sink>
[[nodiscard]] bool AppendVulkanSortClassifySource(
    Sink &sink, const std::string_view key_type)
    noexcept(noexcept(sink.append(std::string_view{}))) {
  VulkanSourceTextSink source{sink};
  source += "layout(set = 0, binding = 0, std430) readonly buffer ";
  source += "SourceKeys {\n  ";
  source += key_type;
  source += " source_keys[];\n};\n";
  source += "layout(set = 0, binding = 1, std430) buffer BlockCounts {\n";
  source += "  uint block_counts[];\n};\n";
  source += "shared uint local_counts[kSortBucketCount];\n";
  source += "void main() {\n";
  source += "  const uint block = sort_dispatch.base_block + "
            "gl_WorkGroupID.x;\n";
  source += "  const uint tid = gl_LocalInvocationID.x;\n";
  source += "  const uint begin = block * kSortBlockSize;\n";
  source += "  const SortRange range = ResolveRange();\n";
  source += "  const uint element_count = uint(range.logical);\n";
  source += "  const uint shift = params.pass_index * 8u;\n";
  source += "  local_counts[tid] = 0u;\n";
  source += "  barrier();\n";
  source += "  for (uint item = 0u; item < kSortItemsPerThread; ++item) {\n";
  source += "    const uint index = begin + tid + item * kSortThreadCount;\n";
  source += "    if (index < element_count) {\n";
  source += "      const uint bucket = rund_sort_ordered_bucket("
            "source_keys[index], shift);\n";
  source += "      atomicAdd(local_counts[bucket], 1u);\n";
  source += "    }\n";
  source += "  }\n";
  source += "  barrier();\n";
  source += "  block_counts[tid * params.block_count + block] = "
            "local_counts[tid];\n";
  source += "}\n";
  return source.ok();
}
#endif

} // namespace rund::node::accel::detail
