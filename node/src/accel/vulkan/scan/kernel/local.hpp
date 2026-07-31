#pragma once

#include "../../../scan/shape.hpp"
#include "../../../scan/vulkan.hpp"
#include "../../adapter/api.hpp"
#include "../../buffer/resident/model.hpp"
#include "../../status.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
struct VulkanKernelScanResources {
  VulkanAdapter *adapter = nullptr;
  rund::kernel::ScanPlan plan{};
  rund::kernel::GraphControl control{};
  VulkanResidentBufferResult input{};
  VulkanResidentBufferResult output{};
  VulkanResidentBufferResult logical_count{};
  VulkanBuffer totals{};
  VulkanStatus status{};
  std::shared_ptr<void> scan_resources{};
};
#endif

} // namespace rund::node::accel::detail
