#include "surface/api.hpp"

#include "test/assert.hpp"

namespace node_accel_contract {

int RunAccelKernelCollectiveSurfaceContract() {
  TEST_ASSERT(AccelGraphKernelCollectiveContract());
  TEST_ASSERT(RequiredMetalRunsInclusiveScan());
  TEST_ASSERT(RequiredVulkanRunsInclusiveScan());
  TEST_ASSERT(RequiredMetalRunsBoundedSort());
  TEST_ASSERT(RequiredVulkanRunsBoundedSort());
  TEST_ASSERT(RequiredMetalRunsGather());
  TEST_ASSERT(RequiredVulkanRunsGather());
  TEST_ASSERT(RequiredMetalRunsHistogram());
  TEST_ASSERT(RequiredVulkanRunsHistogram());
  TEST_ASSERT(AccelGraphKernelPartitionCompileContract());
  TEST_ASSERT(RequiredMetalRunsPartition());
  TEST_ASSERT(RequiredVulkanRunsPartition());
  TEST_ASSERT(RequiredMetalRunsReduce());
  TEST_ASSERT(RequiredVulkanRunsReduce());
  TEST_ASSERT(RequiredMetalRunsScatter());
  TEST_ASSERT(RequiredVulkanRunsScatter());
  TEST_ASSERT(RequiredMetalRunsStencil());
  TEST_ASSERT(RequiredVulkanRunsStencil());
  return 0;
}

} // namespace node_accel_contract
