#pragma once

#include "../local/api.hpp"
#include "../../../sort/block/vulkan.hpp"
#include "../../kernel/source_recipe.hpp"
#include "count.hpp"
namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
[[nodiscard]] inline const char *
VulkanSortKeyType(const rund::kernel::SortKey key) noexcept {
  return key == rund::kernel::SortKey::U64 ? "uint64_t" : "uint";
}

template <typename Sink>
[[nodiscard]] bool AppendVulkanSortBaseSource(
    Sink &sink, const rund::kernel::SortKey key,
    const rund::kernel::u32 local_size, const bool chunked)
    noexcept(noexcept(sink.append(std::string_view{}))) {
  const bool key64 = key == rund::kernel::SortKey::U64;
  const char *const key_type = VulkanSortKeyType(key);
  VulkanSourceTextSink source{sink};
  source += "#version 450\n";
  source += "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : ";
  source += "require\n";
  source += "layout(local_size_x = ";
  source.decimal(local_size);
  source += ") in;\n";
  if (chunked) {
    source += "layout(push_constant) uniform SortDispatch {\n";
    source += "  uint base_block;\n";
    source += "} sort_dispatch;\n";
  }
  source += "const uint kSortThreadCount = ";
  source.decimal(kVulkanSortThreadCount);
  source += "u;\nconst uint kSortItemsPerThread = ";
  source.decimal(kVulkanSortItemsPerThread);
  source += "u;\n";
  source += "const uint kSortBlockSize = ";
  source.decimal(kVulkanSortBlockSize);
  source += "u;\nconst uint kSortBucketCount = ";
  source.decimal(kSortBucketCount);
  source += "u;\nuint rund_sort_bucket(";
  source += key_type;
  source += " key, uint shift) {\n";
  source += key64 ? "  return uint((key >> shift) & uint64_t(255));\n"
                  : "  return (key >> shift) & 255u;\n";
  source += "}\n";
  source += "layout(set = 0, binding = 7, std430) readonly buffer Params {\n";
  source += "  uint64_t element_count;\n";
  source += "  uint block_count;\n";
  source += "  uint pass_index;\n";
  source += "  uint identity_values;\n";
  source += "  uint signed_order;\n";
  source += "  uint pass_count;\n";
  source += "  uint count_words;\n";
  source += "  uint max_dispatch_groups;\n";
  source += "  uint chunk_count;\n";
  source += "} params;\n";
  AppendVulkanSortCount(source);
  source += "uint rund_sort_ordered_bucket(";
  source += key_type;
  source += " key, uint shift) {\n";
  source += "  uint bucket = rund_sort_bucket(key, shift);\n";
  source += "  if (params.signed_order != 0u && ";
  source += "params.pass_index + 1u == params.pass_count) { ";
  source += "bucket ^= 128u; }\n";
  source += "  return bucket;\n}\n";
  return source.ok();
}
#endif
} // namespace rund::node::accel::detail
