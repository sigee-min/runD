#include <accel/api.hpp>
#include <accel/device.hpp>

#include "reduce/local.hpp"
#include "reduce/match/run.hpp"
#include "reduce/reject/run.hpp"
#include "src/accel/metal/reduce/local.hpp"
#include "src/accel/vulkan/reduce/local.hpp"
#include <node/accel/pick.hpp>
#include <string>
namespace node_accel_contract {
namespace {
#include "reduce/source.hpp"

} // namespace

bool BackendRunsReduce(const rund::AccelDevice &pick) {
  return SignedReduceSourcesCarryDomainOrder() &&
         WideReduceSourcesCarryFixedHierarchy() && reduce::MatchesU32(pick) &&
         reduce::MatchesU64(pick) && reduce::CountsNonzeroU32(pick) &&
         reduce::CountsNonzeroU64(pick) &&
         reduce::MatchesWideHierarchyU32(pick) &&
         reduce::MatchesWideHierarchyI64Cancellation(pick) &&
         reduce::RejectsWideU32Overflow(pick) &&
         reduce::RejectsWideU64Overflow(pick) &&
         reduce::MatchesMinU32(pick) &&
         reduce::MatchesMaxU64(pick) &&
         reduce::MatchesMinU32NonPowerOfTwoBlock(pick) &&
         reduce::MatchesMaxU64NonPowerOfTwoBlock(pick) &&
         reduce::MatchesMinI32(pick) && reduce::RejectsU32Overflow(pick);
}

bool RequiredMetalRunsReduce() {
  const rund::AccelDevice pick =
      rund::node::accel::PickAccel(primitive::Policy(rund::AccelApi::Metal));
  if (!pick.check.ok) {
    return primitive::PickUnavailableReasonIsPrecise(pick,
                                                     rund::AccelApi::Metal);
  }
  return pick.api == rund::AccelApi::Metal && BackendRunsReduce(pick);
}

bool RequiredVulkanRunsReduce() {
  const rund::AccelDevice pick =
      rund::node::accel::PickAccel(primitive::Policy(rund::AccelApi::Vulkan));
  if (!pick.check.ok) {
    return primitive::PickUnavailableReasonIsPrecise(pick,
                                                     rund::AccelApi::Vulkan);
  }
  return pick.api == rund::AccelApi::Vulkan && BackendRunsReduce(pick);
}

} // namespace node_accel_contract
