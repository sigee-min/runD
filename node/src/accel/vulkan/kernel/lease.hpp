#pragma once

#include "../adapter/state.hpp"

#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

class VulkanLeaseScope final {
public:
  VulkanLeaseScope(
      VulkanAdapter &adapter,
      std::vector<VulkanCollectiveDescriptorLease> &leases) noexcept
      : adapter_{adapter}, previous_{adapter.active_descriptor_leases} {
    adapter_.active_descriptor_leases = &leases;
  }

  ~VulkanLeaseScope() { adapter_.active_descriptor_leases = previous_; }

  VulkanLeaseScope(const VulkanLeaseScope &) = delete;
  VulkanLeaseScope &operator=(const VulkanLeaseScope &) = delete;

private:
  VulkanAdapter &adapter_;
  std::vector<VulkanCollectiveDescriptorLease> *previous_{};
};

inline void ReleaseVulkanLeases(
    std::vector<VulkanCollectiveDescriptorLease> &leases) noexcept {
  for (const VulkanCollectiveDescriptorLease lease : leases) {
    if (lease.pipeline != nullptr &&
        lease.slot < lease.pipeline->descriptor_leased.size()) {
      lease.pipeline->descriptor_leased[lease.slot] = false;
    }
  }
  leases.clear();
}

#endif

} // namespace rund::node::accel::detail
