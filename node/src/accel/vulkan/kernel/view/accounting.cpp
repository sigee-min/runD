#include "../view.hpp"

#include <rund/counter.hpp>

#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

PreparedMemory VulkanViewMemory(const std::shared_ptr<VulkanViewLowering> &view,
                                const std::uint64_t budget,
                                std::uint64_t &traffic) noexcept {
  std::uint64_t bytes = 0u;
  traffic = 0u;
  if (view != nullptr) {
    for (const VulkanViewTransfer &transfer : view->transfers) {
      const std::uint64_t dense = transfer.count * transfer.element_bytes;
      traffic = ::rund::detail::counter::SaturatingAdd(traffic, dense);
      if (transfer.planned) {
        continue;
      }
      bytes = bytes > std::numeric_limits<std::uint64_t>::max() - dense
                  ? std::numeric_limits<std::uint64_t>::max()
                  : bytes + dense;
    }
  }
  return PreparedMemory{
      .current = bytes, .peak = bytes, .cumulative = bytes, .budget = budget};
}

std::uint64_t VulkanViewDispatchCount(
    const std::shared_ptr<VulkanViewLowering> &view) noexcept {
  std::uint64_t count = 0u;
  if (view != nullptr) {
    for (const VulkanViewTransfer &transfer : view->transfers) {
      count = ::rund::detail::counter::SaturatingAdd(
          count, static_cast<std::uint64_t>(transfer.pages.size()));
    }
  }
  return count;
}

#endif

} // namespace rund::node::accel::detail
