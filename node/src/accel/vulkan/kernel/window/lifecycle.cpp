#include "../../buffer/access.hpp"

#include "internal.hpp"

#include "../copy.hpp"
#include "../lease.hpp"

#include "../../../kernel/footprint.hpp"

#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

void DestroyVulkanWindow(VulkanWindowResources &resources) noexcept {
  if (resources.adapter != nullptr) {
    ReleaseVulkanLeases(resources.descriptor_leases);
    DestroyVulkanBuffer(*resources.adapter, resources.states);
    DestroyVulkanBuffer(*resources.adapter, resources.arguments);
    DestroyVulkanBuffer(*resources.adapter, resources.original_arguments);
    DestroyVulkanBuffer(*resources.adapter, resources.owners);
  }
  resources = {};
}

bool FreezeVulkanWindow(VulkanWindowResources &resources) noexcept {
  const std::uint64_t argument_bytes =
      static_cast<std::uint64_t>(resources.original.size()) *
      sizeof(VkDispatchIndirectCommand);
  return resources.state_count == 0u ||
         (argument_bytes == resources.original_arguments.bytes &&
          UploadVulkanBuffer(resources.original_arguments,
                             resources.original.data(), argument_bytes));
}

std::uint64_t
VulkanWindowHostBytes(const VulkanWindowResources &resources) noexcept {
  const std::uint64_t routes = capacity_bytes(resources.routes);
  const std::uint64_t leases = capacity_bytes(resources.descriptor_leases);
  const std::uint64_t original = capacity_bytes(resources.original);
  const std::uint64_t gates = capacity_bytes(resources.gates);
  constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  const std::uint64_t base =
      routes > maximum - leases ? maximum : routes + leases;
  const std::uint64_t captured =
      base > maximum - original ? maximum : base + original;
  return captured > maximum - gates ? maximum : captured + gates;
}

#endif

} // namespace rund::node::accel::detail
