#include <accel/check.hpp>

#include "../../kernel/finish.hpp"
#include "local.hpp"
#include "ops/table.hpp"

#include <algorithm>
#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck FinishVulkanStep(VulkanAdapter &adapter,
                                  const VulkanKernelOps &ops,
                                  const std::shared_ptr<void> &resources) {
  if (ops.finish != nullptr) {
    return ops.finish(adapter, resources);
  }
  return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
}

rund::AccelCheck FinishVulkanSteps(VulkanAdapter &adapter,
                                   VulkanKernelResources &resources,
                                   rund::RuntimeStats *const stats) {
  std::size_t failed = std::numeric_limits<std::size_t>::max();
  const rund::AccelCheck result = finish::Steps(
      resources,
      [&](const VulkanKernelEntry &entry) {
        return FinishVulkanStep(adapter, entry.ops, entry.resource);
      },
      &failed);
  if (!result.ok && stats != nullptr && failed < resources.size()) {
    const VulkanKernelEntry *const entry = resources.entry(failed);
    std::uint64_t ordinal = std::numeric_limits<std::uint64_t>::max();
    if (entry != nullptr && entry->ops.failure != nullptr &&
        entry->ops.failure(entry->resource, ordinal)) {
      stats->overflow_ordinal = std::min(stats->overflow_ordinal, ordinal);
    }
  }
  if (result.ok) {
    SetVulkanLastError(adapter, "ok");
  }
  return result;
}

#endif

} // namespace rund::node::accel::detail
