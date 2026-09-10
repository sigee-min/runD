#include "local.hpp"

#include "../../buffer/create/telemetry.hpp"
#include "../view.hpp"

#include <rund/counter.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

void ProjectVulkanPreparedMemory(const VulkanMemoryStats before,
                                 const VulkanAdapter &adapter,
                                 VulkanKernelResources &resources,
                                 PreparedMemory &memory) {
  memory = VulkanPreparedMemory(before, adapter.staging_memory,
                                adapter.caps.staging_bytes);
  for (std::size_t index = 0u; index < resources.size(); ++index) {
    const VulkanKernelEntry *const entry = resources.entry(index);
    if (entry != nullptr) {
      std::uint64_t traffic = 0u;
      accumulate_memory(
          memory,
          VulkanViewMemory(entry->view, adapter.caps.staging_bytes, traffic));
      resources.traffic =
          ::rund::detail::counter::SaturatingAdd(resources.traffic, traffic);
    }
  }
}

#endif

} // namespace rund::node::accel::detail
