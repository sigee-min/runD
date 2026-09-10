#include "local.hpp"

#include "../../metal/stats/run.hpp"
#include "../../vulkan/stats/run.hpp"
#include "test/assert.hpp"

namespace node_accel_contract::backend_runtime {

bool CheckMetalRuntime(const rund::AccelDevice &pick) {
  if (!pick.check.ok) {
    return node_accel_contract::MetalFailsClosed(pick);
  }
  TEST_ASSERT(
      node_accel_contract::MetalRepeatedStagedRunsReportWarmRuntimeStats(pick));
  TEST_ASSERT(
      node_accel_contract::MetalResidentBufferRegistryValidatesPrivateRefs(
          pick));
  TEST_ASSERT(node_accel_contract::PublicBufferApiContract(pick));
  return true;
}

bool CheckVulkanRuntime() {
  TEST_ASSERT(
      node_accel_contract::VulkanRepeatedStagedRunsReportWarmRuntimeStats());
  return true;
}

} // namespace node_accel_contract::backend_runtime
