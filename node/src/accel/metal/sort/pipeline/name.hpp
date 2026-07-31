#pragma once

#include "../local.hpp"
#include "../../../sort/block/metal.hpp"
#include "../../pipeline/named.hpp"

#include <string>
#include <string_view>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
inline void AppendSortShape(std::string &name,
                            const rund::kernel::u32 block_size) {
  name += ".b";
  name += std::to_string(block_size);
  name += ".t";
  name += std::to_string(kMetalSortThreadCount);
  name += ".r";
  name += std::to_string(kMetalSortRounds);
  name += ".w";
  name += std::to_string(kMetalSortSimdWidth);
  name += ".bucketmajor.staged";
}

[[nodiscard]] inline NSString* SortFunctionName(
    const char* const base,
    const rund::kernel::SortKey key) {
  std::string name = base;
  name += key == rund::kernel::SortKey::U64 ? "_u64" : "_u32";
  return [NSString stringWithUTF8String:name.c_str()];
}

[[nodiscard]] inline std::string SortPipelineKey(
    const std::string_view stage,
    const rund::kernel::SortKey key,
    const rund::kernel::u32 block_size) {
  std::string name = "sort.";
  name += stage;
  name += key == rund::kernel::SortKey::U64 ? ".u64" : ".u32";
  AppendSortShape(name, block_size);
  return name;
}

[[nodiscard]] inline std::string
SortPipelineKey(const std::string_view stage,
                const rund::kernel::u32 block_size) {
  std::string name = "sort.";
  name += stage;
  AppendSortShape(name, block_size);
  return name;
}
#endif

}  // namespace rund::node::accel::detail
