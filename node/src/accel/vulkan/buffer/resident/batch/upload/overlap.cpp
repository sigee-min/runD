#include "../local.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool overlaps(const std::span<const UploadPlan> plans) {
  if (plans.size() < 2u) {
    return false;
  }
  std::vector<const UploadPlan *> ordered;
  ordered.reserve(plans.size());
  for (const UploadPlan &plan : plans) {
    ordered.push_back(&plan);
  }
  std::sort(ordered.begin(), ordered.end(),
            [](const UploadPlan *const lhs, const UploadPlan *const rhs) {
              if (lhs->resident != rhs->resident) {
                return std::less<const VulkanBuffer *>{}(lhs->resident,
                                                         rhs->resident);
              }
              return lhs->range.offset < rhs->range.offset;
            });
  const VulkanBuffer *resident = nullptr;
  VkDeviceSize end = 0u;
  for (const UploadPlan *const plan : ordered) {
    if (plan->resident != resident) {
      resident = plan->resident;
      end = plan->range.offset + plan->range.bytes;
      continue;
    }
    if (plan->range.offset < end) {
      return true;
    }
    end = std::max(end, plan->range.offset + plan->range.bytes);
  }
  return false;
}

#endif

} // namespace rund::node::accel::detail
