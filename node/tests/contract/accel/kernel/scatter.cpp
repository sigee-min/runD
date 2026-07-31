#include <accel/api.hpp>
#include <accel/device.hpp>

#include <kernel/program/compute/scatter/plan.hpp>

#include "scatter/local.hpp"
#include "scatter/match/run.hpp"
#include "scatter/reject/reduce.hpp"
#include "scatter/reject/run.hpp"
#include "src/accel/scatter/shape.hpp"
#include <node/accel/pick.hpp>

namespace node_accel_contract {

namespace {

[[nodiscard]] bool ScatterEncodingBoundary() noexcept {
  constexpr rund::kernel::u64 max_count = 0x7fffffffu;
  const rund::kernel::ScatterDesc admitted{
      .element = rund::kernel::ScatterElement::U32,
      .element_count = max_count,
      .output_count = 1u,
  };
  const rund::kernel::ScatterPlan admitted_plan =
      rund::kernel::PlanScatter(admitted);
  const rund::kernel::ScatterDesc rejected{
      .element = rund::kernel::ScatterElement::U32,
      .element_count = max_count + 1u,
      .output_count = 1u,
  };
  const rund::kernel::ScatterPlan rejected_plan =
      rund::kernel::PlanScatter(rejected);
  return rund::kernel::ScatterPlanMatchesDesc(admitted, admitted_plan) &&
         !rund::kernel::ScatterPlanMatchesDesc(rejected, rejected_plan);
}

} // namespace

namespace scatter {

bool ScatterReduceFailuresAreAtomic(const rund::AccelDevice &pick) {
  return reject::ScatterReduceFailuresAreAtomic(pick);
}

bool ScatterReduceParallelModes(const rund::AccelDevice &pick) {
  return reject::ScatterReduceParallelModes(pick);
}

} // namespace scatter

bool BackendRunsScatter(const rund::AccelDevice &pick) {
  return ScatterEncodingBoundary() && scatter::MatchesU32(pick) &&
         scatter::MatchesU64(pick) && scatter::RejectsDuplicateIndex(pick) &&
         scatter::ScatterReduceFailuresAreAtomic(pick) &&
         scatter::ScatterReduceParallelModes(pick);
}

bool RequiredMetalRunsScatter() {
  const rund::AccelDevice pick =
      rund::node::accel::PickAccel(primitive::Policy(rund::AccelApi::Metal));
  if (!pick.check.ok) {
    return primitive::PickUnavailableReasonIsPrecise(pick,
                                                     rund::AccelApi::Metal);
  }
  return pick.api == rund::AccelApi::Metal && BackendRunsScatter(pick);
}

bool RequiredVulkanRunsScatter() {
  const rund::AccelDevice pick =
      rund::node::accel::PickAccel(primitive::Policy(rund::AccelApi::Vulkan));
  if (!pick.check.ok) {
    return primitive::PickUnavailableReasonIsPrecise(pick,
                                                     rund::AccelApi::Vulkan);
  }
  return pick.api == rund::AccelApi::Vulkan && BackendRunsScatter(pick);
}

} // namespace node_accel_contract
