#include <accel/device.hpp>

#include "partition/local.hpp"
#include <node/accel/pick.hpp>

namespace node_accel_contract {

bool AccelGraphKernelPartitionCompileContract() {
  return partition::CompileContract();
}

bool BackendRunsPartition(const rund::AccelDevice &pick) {
  return pick.check.ok && partition::MatchesReference(pick);
}

namespace {

[[nodiscard]] bool RequiredPartition(const rund::AccelApi api) {
  const rund::AccelDevice pick =
      rund::node::accel::PickAccel(primitive::Policy(api));
  if (!pick.check.ok) {
    return primitive::PickUnavailableReasonIsPrecise(pick, api);
  }
  return pick.api == api && BackendRunsPartition(pick);
}

} // namespace

bool RequiredMetalRunsPartition() {
  return RequiredPartition(rund::AccelApi::Metal);
}

bool RequiredVulkanRunsPartition() {
  return RequiredPartition(rund::AccelApi::Vulkan);
}

} // namespace node_accel_contract
