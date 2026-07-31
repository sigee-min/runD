#include <accel/api.hpp>
#include <accel/device.hpp>

#include "histogram/local.hpp"
#include <node/accel/pick.hpp>

namespace node_accel_contract {

bool BackendRunsHistogram(const rund::AccelDevice &pick) {
  return histogram::MatchesU32(pick) && histogram::RejectsOutOfRangeBin(pick);
}

bool RequiredMetalRunsHistogram() {
  const rund::AccelDevice pick =
      rund::node::accel::PickAccel(primitive::Policy(rund::AccelApi::Metal));
  if (!pick.check.ok) {
    return primitive::PickUnavailableReasonIsPrecise(pick,
                                                     rund::AccelApi::Metal);
  }
  return pick.api == rund::AccelApi::Metal && BackendRunsHistogram(pick);
}

bool RequiredVulkanRunsHistogram() {
  const rund::AccelDevice pick =
      rund::node::accel::PickAccel(primitive::Policy(rund::AccelApi::Vulkan));
  if (!pick.check.ok) {
    return primitive::PickUnavailableReasonIsPrecise(pick,
                                                     rund::AccelApi::Vulkan);
  }
  return pick.api == rund::AccelApi::Vulkan && BackendRunsHistogram(pick);
}

} // namespace node_accel_contract
