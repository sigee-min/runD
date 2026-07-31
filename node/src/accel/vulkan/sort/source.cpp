#include "source/classify.hpp"
#include "source/dispatch.hpp"
#include "source/prefix.hpp"
#include "source/scatter.hpp"
namespace rund::node::accel::detail {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
std::string VulkanSortSource(const rund::kernel::SortKey key,
                             const SortStage stage) {
  const rund::kernel::u32 local_size =
      stage == SortStage::Dispatch ? 1u : kVulkanSortThreadCount;
  const bool chunked =
      stage == SortStage::Classify || stage == SortStage::Scatter;
  std::string source = VulkanSortBaseSource(key, local_size, chunked);
  const std::string key_type = VulkanSortKeyType(key);
  switch (stage) {
  case SortStage::Dispatch:
    source += VulkanSortDispatchSource();
    break;
  case SortStage::Classify:
    source += VulkanSortClassifySource(key_type);
    break;
  case SortStage::Prefix:
    source += VulkanSortPrefixSource();
    break;
  case SortStage::Base:
    source += VulkanSortBaseOffsetSource();
    break;
  case SortStage::Scatter:
    source += VulkanSortScatterSource(key_type);
    break;
  }
  return source;
}
#endif
} // namespace rund::node::accel::detail
