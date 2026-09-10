#include "local.hpp"

#include "test/assert.hpp"

namespace node_accel_contract {
[[nodiscard]] bool ResidentValidationContract();
}

int RunAccelBackendRuntimeContract() {
  TEST_ASSERT(node_accel_contract::backend_runtime::CheckPickTokenAdmission());
  TEST_ASSERT(node_accel_contract::ResidentValidationContract());
  TEST_ASSERT(
      node_accel_contract::backend_runtime::CheckVulkanCommandFailure());
  TEST_ASSERT(node_accel_contract::backend_runtime::CheckCpuCounters(
      node_accel_contract::backend_runtime::Pick(rund::AccelApi::Cpu)));
  TEST_ASSERT(node_accel_contract::backend_runtime::CheckMetalCounters(
      node_accel_contract::backend_runtime::Pick(rund::AccelApi::Metal)));
  TEST_ASSERT(node_accel_contract::backend_runtime::CheckVulkanCounters(
      node_accel_contract::backend_runtime::Pick(rund::AccelApi::Vulkan)));
  TEST_ASSERT(node_accel_contract::backend_runtime::CheckMetalHostReadback());
  TEST_ASSERT(node_accel_contract::backend_runtime::CheckVulkanHostReadback());
  TEST_ASSERT(node_accel_contract::backend_runtime::CheckVulkanMemoryTier(
      node_accel_contract::backend_runtime::Pick(rund::AccelApi::Vulkan)));
  TEST_ASSERT(node_accel_contract::backend_runtime::CheckMetalRuntime(
      node_accel_contract::backend_runtime::Pick(rund::AccelApi::Metal)));
  TEST_ASSERT(node_accel_contract::backend_runtime::CheckVulkanRuntime());
  return 0;
}
