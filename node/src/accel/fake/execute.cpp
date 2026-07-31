#include "local.hpp"

#include "../plan/validation.hpp"
#include "../resident/window/admission/runtime/windows.hpp"

#include <kernel/program/compute/lowering/artifact/admission.hpp>
#include <kernel/program/compute/plan.hpp>
namespace rund::node::accel::detail {

bool ExecuteFake(void *const context, const rund::kernel::ComputePlan &plan,
                 const rund::kernel::LoweringArtifact &artifact,
                 const rund::kernel::ComputeDispatchWindow *windows,
                 const rund::kernel::u64 window_count,
                 const rund::kernel::BindingSet &bindings) {
  const auto *const adapter = static_cast<const FakeAdapter *>(context);
  if (adapter == nullptr || !plan.ok ||
      !rund::kernel::ComputePlanBytesValid(plan) ||
      !FrozenCapsAdmitPlan(adapter->caps, plan) ||
      !rund::kernel::compute_lowering_detail::AdmitArtifact(plan, artifact)
           .ok ||
      window_count != plan.dispatch_count) {
    return false;
  }

  const rund::kernel::BindingValidation validation =
      BindingPlanCheck(plan, bindings, artifact.metadata,
                       PlanBindingInputMode::StagedOrResident);
  return validation.ok &&
         RuntimeWindowsMatchPlan(plan, windows, window_count, bindings);
}

bool ExecuteRetainedFake(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings) {
  const auto *const adapter = static_cast<const FakeAdapter *>(
      pick.api == rund::AccelApi::Fake ? pick.backend.context : nullptr);
  const auto admission = rund::kernel::compute_lowering_detail::AdmitRetained(
      plan, artifact, nullptr);
  if (adapter == nullptr || !plan.ok ||
      !rund::kernel::ComputePlanBytesValid(plan) ||
      !FrozenCapsAdmitPlan(adapter->caps, plan) || !admission.ok ||
      admission.parse_count != 0u || admission.emission_count != 0u ||
      window_count != plan.dispatch_count) {
    return false;
  }
  const rund::kernel::BindingValidation validation =
      BindingPlanCheck(plan, bindings, artifact.metadata,
                       PlanBindingInputMode::StagedOrResident);
  return validation.ok &&
         RuntimeWindowsMatchPlan(plan, windows, window_count, bindings);
}

} // namespace rund::node::accel::detail
