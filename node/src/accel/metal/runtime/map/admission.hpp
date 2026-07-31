#pragma once

#include <accel/check.hpp>

#include "../local.hpp"

#include <kernel/program/compute/lowering/artifact/admission.hpp>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] rund::AccelCheck ValidateMetalMapPrepare(
    MetalAdapter &adapter, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings) {
  if (!ComputePlanHeaderValid(plan, rund::kernel::ComputeApi::Metal)) {
    SetMetalLastError(adapter, "compute_plan_invalid");
    return rund::AccelCheck{false, "compute_plan_invalid"};
  }
  if (!FrozenCapsAdmitPlan(adapter.caps, plan)) {
    SetMetalLastError(adapter, "compute_backend_mismatch");
    return rund::AccelCheck{false, "compute_backend_mismatch"};
  }
  const auto retained =
      rund::kernel::compute_lowering_detail::AdmitRetained(
          plan, artifact, nullptr);
  if (!retained.ok || retained.parse_count != 0u ||
      retained.emission_count != 0u) {
    SetMetalLastError(adapter, "compute_artifact_mismatch");
    return rund::AccelCheck{false, "compute_artifact_mismatch"};
  }
  const rund::kernel::BindingValidation binding =
      BindingPlanCheck(plan, bindings, artifact.metadata,
                       PlanBindingInputMode::MapResidentViews);
  if (!binding.ok) {
    SetMetalLastError(adapter, binding.reason);
    return rund::AccelCheck{false, binding.reason};
  }
  if (!RuntimeWindowsMatchPlan(plan, windows, window_count, bindings) ||
      !bindings.has_resident_output()) {
    SetMetalLastError(adapter, "compute_dispatch_count_mismatch");
    return rund::AccelCheck{false, "compute_dispatch_count_mismatch"};
  }
  return rund::AccelCheck{true, "ok"};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
