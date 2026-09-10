#include "../publish.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#include "../../../kernel/footprint.hpp"

#include <limits>

namespace rund::node::accel::detail {

std::uint64_t VulkanPipelinePublishHostBytes(
    const VulkanPipelinePublishResources &resources) noexcept {
  const std::uint64_t routes = capacity_bytes(resources.routes);
  const std::uint64_t leases = capacity_bytes(resources.descriptor_leases);
  constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  return routes > maximum - leases ? maximum : routes + leases;
}

} // namespace rund::node::accel::detail

#endif
