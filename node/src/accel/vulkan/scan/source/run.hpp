#pragma once

#include "offset.hpp"

namespace rund::node::accel::detail {

std::string VulkanScanSource(const rund::kernel::ScanElement element,
                             const rund::kernel::ComputeDomain domain,
                             const VulkanScanStage stage, const bool inclusive) {
  std::string source;
  AppendVulkanScanPrelude(source, element, domain, kVulkanScanWidth,
                          stage != VulkanScanStage::Prefix);
  switch (stage) {
  case VulkanScanStage::Block:
    AppendVulkanScanBlock(source, element, inclusive);
    break;
  case VulkanScanStage::Prefix:
    AppendVulkanScanPrefix(source, element);
    break;
  case VulkanScanStage::Offset:
    AppendVulkanScanOffset(source, element);
    break;
  }
  return source;
}

} // namespace rund::node::accel::detail
