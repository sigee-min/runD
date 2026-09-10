#include "../record.hpp"

#include "../../../../../kernel/prepared/template/registry.hpp"
#include "../../../../command/resources.hpp"

#include <kernel/core/checked.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck
DescribeVulkanRouteDispatches(const VulkanKernelResources &resources,
                              std::uint64_t &total,
                              std::uint64_t &indirect) noexcept {
  total = 0u;
  indirect = 0u;
  if (resources.adapter == nullptr ||
      resources.adapter->max_dispatch_groups == 0u ||
      resources.program == nullptr ||
      resources.program->steps.size() != resources.size()) {
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }
  const std::uint64_t reset_window =
      static_cast<std::uint64_t>(resources.adapter->max_dispatch_groups) * 256u;
  for (const VulkanReset &clear : resources.resets) {
    if (!clear.shader ||
        !rund::kernel::checked::add(
            total, reset::Commands(clear.range.count(), reset_window), total)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  for (std::size_t index = 0u; index < resources.size(); ++index) {
    const VulkanKernelEntry *const entry = resources.entry(index);
    const PreparedBackendManifest &manifest =
        resources.program->steps[index].manifest;
    std::uint64_t step_total = 0u;
    std::uint64_t encoded_indirect = 0u;
    if (entry == nullptr || !manifest.ok ||
        !rund::kernel::checked::add(manifest.capture_direct_dispatch_count,
                                    manifest.capture_indirect_dispatch_count,
                                    step_total)) {
      return rund::AccelCheck{false, "accel_kernel_template_invalid"};
    }
    if (entry->ops.pipeline_capture_demand != nullptr) {
      const rund::AccelCheck described =
          entry->ops.pipeline_capture_demand(entry->resource, encoded_indirect);
      if (!described.ok) {
        return described;
      }
    }
    if (encoded_indirect != manifest.capture_indirect_dispatch_count ||
        !rund::kernel::checked::add(total, step_total, total) ||
        !rund::kernel::checked::add(total, VulkanViewDispatchCount(entry->view),
                                    total) ||
        !rund::kernel::checked::add(indirect, encoded_indirect, indirect)) {
      return rund::AccelCheck{false, "compute_dispatch_count_mismatch"};
    }
  }
  return total == 0u
             ? rund::AccelCheck{false, "compute_dispatch_count_mismatch"}
             : rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
