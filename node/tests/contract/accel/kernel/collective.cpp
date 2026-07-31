#include <accel/device.hpp>

#include "collective/local.hpp"

namespace node_accel_contract {

bool AccelGraphKernelCollectiveContract() {
  return collective::GraphContract();
}

bool RequiredMetalRunsAccelGraphScan(const rund::AccelDevice &pick) {
  return collective::RequiredMetalScan(pick);
}

bool RequiredVulkanRunsAccelGraphScan(const rund::AccelDevice &pick) {
  return collective::RequiredVulkanScan(pick);
}

bool BackendRunsScanSortCompact(const rund::AccelDevice &pick) {
  return collective::BackendRunsScanSortCompact(pick);
}

bool AvailableBackendsRunSameAccelGraphSort(const rund::AccelDevice &metal,
                                            const rund::AccelDevice &vulkan) {
  return collective::AvailableBackendsRunSameSort(metal, vulkan);
}

} // namespace node_accel_contract
