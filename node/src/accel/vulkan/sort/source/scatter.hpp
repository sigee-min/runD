#pragma once

#include "base.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
template <typename Sink>
[[nodiscard]] bool AppendVulkanSortScatterSource(
    Sink &sink, const std::string_view key_type)
    noexcept(noexcept(sink.append(std::string_view{}))) {
  VulkanSourceTextSink source{sink};
  source += "layout(set = 0, binding = 0, std430) readonly buffer ";
  source += "SourceKeys {\n  ";
  source += key_type;
  source += " source_keys[];\n};\n";
  source += "layout(set = 0, binding = 1, std430) readonly buffer ";
  source += "SourceValues {\n  uint source_values[];\n};\n";
  source += "layout(set = 0, binding = 2, std430) buffer TargetKeys {\n  ";
  source += key_type;
  source += " target_keys[];\n};\n";
  source += "layout(set = 0, binding = 3, std430) buffer ";
  source += "TargetValues {\n  uint target_values[];\n};\n";
  source += "layout(set = 0, binding = 5, std430) readonly buffer ";
  source += "BlockOffsets {\n  uint block_offsets[];\n};\n";
  source += "layout(set = 0, binding = 6, std430) readonly buffer ";
  source += "BlockCounts {\n  uint block_counts[];\n};\n";
  source += "const uint kSortRankGroupSize = ";
  source.decimal(kVulkanSortRankGroupSize);
  source += "u;\nconst uint kSortRankGroupCount = ";
  source.decimal(kVulkanSortRankGroupCount);
  source += "u;\nconst uint kSortGroupsPerWord = ";
  source.decimal(kVulkanSortGroupsPerWord);
  source += "u;\nconst uint kSortPackedWordCount = ";
  source.decimal(kVulkanSortPackedWordCount);
  source += "u;\n";
  source += "shared uint local_buckets[kSortBlockSize];\n";
  source += "shared uint packed_counts["
            "kSortBucketCount * kSortPackedWordCount];\n";
  source += "uint rund_sort_word_sum(uint word) {\n";
  source += "  word = (word & 0x0f0f0f0fu) + "
            "((word >> 4u) & 0x0f0f0f0fu);\n";
  source += "  word = (word & 0x00ff00ffu) + "
            "((word >> 8u) & 0x00ff00ffu);\n";
  source += "  return (word & 0xffffu) + (word >> 16u);\n";
  source += "}\n";
  source += "uint rund_sort_group_prefix(uint bucket, uint group) {\n";
  source += "  const uint word = group / kSortGroupsPerWord;\n";
  source += "  uint prefix = 0u;\n";
  source += "  for (uint prior = 0u; prior < word; ++prior) {\n";
  source += "    prefix += rund_sort_word_sum(packed_counts["
            "bucket * kSortPackedWordCount + prior]);\n";
  source += "  }\n";
  source += "  const uint shift = "
            "(group & (kSortGroupsPerWord - 1u)) * 4u;\n";
  source += "  const uint mask = shift == 0u ? 0u : (1u << shift) - 1u;\n";
  source += "  prefix += rund_sort_word_sum(packed_counts["
            "bucket * kSortPackedWordCount + word] & mask);\n";
  source += "  return prefix;\n";
  source += "}\n";
  source += "void main() {\n";
  source += "  const uint tid = gl_LocalInvocationID.x;\n";
  source += "  const SortRange range = ResolveRange();\n";
  source += "  const uint block = sort_dispatch.base_block + "
            "gl_WorkGroupID.x;\n";
  source += "  const uint begin = block * kSortBlockSize;\n";
  source += "  const uint shift = params.pass_index * 8u;\n";
  source += "  ";
  source += key_type;
  source += " keys[kSortItemsPerThread];\n";
  source += "  uint buckets[kSortItemsPerThread];\n";
  source += "  uint ranks[kSortItemsPerThread];\n";
  source += "  bool enabled[kSortItemsPerThread];\n";
  source += "  for (uint entry = tid; entry < "
            "kSortBucketCount * kSortPackedWordCount; "
            "entry += kSortThreadCount) {\n";
  source += "    packed_counts[entry] = 0u;\n";
  source += "  }\n";
  source += "  for (uint item = 0u; item < kSortItemsPerThread; ++item) {\n";
  source += "    const uint local_index = tid + item * kSortThreadCount;\n";
  source += "    const uint index = begin + local_index;\n";
  source += "    enabled[item] = index < uint(range.logical);\n";
  source += "    keys[item] = enabled[item] ? source_keys[index] : ";
  source += key_type == "uint64_t" ? "uint64_t(0)" : "0u";
  source += ";\n";
  source += "    buckets[item] = enabled[item] "
            "? rund_sort_ordered_bucket(keys[item], shift) : 0u;\n";
  source += "    local_buckets[local_index] = buckets[item];\n";
  source += "  }\n";
  source += "  barrier();\n";
  source += "  for (uint item = 0u; item < kSortItemsPerThread; ++item) {\n";
  source += "    const uint local_index = tid + item * kSortThreadCount;\n";
  source += "    const uint group = local_index / kSortRankGroupSize;\n";
  source += "    const uint group_begin = group * kSortRankGroupSize;\n";
  source += "    uint rank = 0u;\n";
  source += "    if (enabled[item]) {\n";
  source +=
      "      for (uint prior = group_begin; prior < local_index; ++prior) {\n";
  source +=
      "        rank += local_buckets[prior] == buckets[item] ? 1u : 0u;\n";
  source += "      }\n";
  source += "      const uint word = group / kSortGroupsPerWord;\n";
  source += "      const uint shift = "
            "(group & (kSortGroupsPerWord - 1u)) * 4u;\n";
  source += "      atomicAdd(packed_counts[buckets[item] * "
            "kSortPackedWordCount + word], 1u << shift);\n";
  source += "    }\n";
  source += "    ranks[item] = rank;\n";
  source += "  }\n";
  source += "  barrier();\n";
  source += "  for (uint item = 0u; item < kSortItemsPerThread; ++item) {\n";
  source += "    const uint local_index = tid + item * kSortThreadCount;\n";
  source += "    const uint group = local_index / kSortRankGroupSize;\n";
  source +=
      "    ranks[item] += rund_sort_group_prefix(buckets[item], group);\n";
  source += "  }\n";
  source += "  barrier();\n";
  source += "  const uint totals = params.block_count * kSortBucketCount;\n";
  source += "  packed_counts[tid] = block_counts[totals + tid];\n";
  source += "  packed_counts[kSortBucketCount + tid] = "
            "params.block_count == 1u || begin >= uint(range.logical) ? 0u : "
            "block_offsets[tid * params.block_count + block];\n";
  source += "  barrier();\n";
  source += "  for (uint item = 0u; item < kSortItemsPerThread; ++item) {\n";
  source += "    if (enabled[item]) {\n";
  source += "      const uint local_index = tid + item * kSortThreadCount;\n";
  source += "      const uint index = begin + local_index;\n";
  source += "      const uint bucket = buckets[item];\n";
  source += "      const uint offset = packed_counts[bucket] + "
            "packed_counts[kSortBucketCount + bucket];\n";
  source += "      const uint target = offset + ranks[item];\n";
  source += "      target_keys[target] = keys[item];\n";
  source += "      target_values[target] = ";
  source += "params.identity_values != 0u && params.pass_index == 0u ";
  source += "? index : source_values[index];\n";
  source += "    }\n";
  source += "  }\n";
  source += "}\n";
  return source.ok();
}
#endif

} // namespace rund::node::accel::detail
