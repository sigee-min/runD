#include "local.hpp"

#include <kernel/program/compute/lowering/artifact/admission.hpp>

#include <array>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
bool ExecuteMetal(void *const context, const rund::kernel::ComputePlan &plan,
                  const rund::kernel::LoweringArtifact &artifact,
                  const rund::kernel::ComputeDispatchWindow *const windows,
                  const rund::kernel::u64 window_count,
                  const rund::kernel::BindingSet &bindings) {
  @autoreleasepool {
    auto *const adapter = static_cast<MetalAdapter *>(context);
    if (adapter == nullptr || adapter->device == nullptr ||
        adapter->queue == nullptr) {
      return false;
    }
    SetMetalLastError(*adapter, "ok");
    if (!ComputePlanHeaderValid(plan, rund::kernel::ComputeApi::Metal)) {
      SetMetalLastError(*adapter, "compute_plan_invalid");
      return false;
    }
    if (!FrozenCapsAdmitPlan(adapter->caps, plan)) {
      SetMetalLastError(*adapter, "compute_backend_mismatch");
      return false;
    }
    if (!rund::kernel::compute_lowering_detail::AdmitArtifact(plan, artifact)
             .ok) {
      SetMetalLastError(*adapter, "compute_artifact_mismatch");
      return false;
    }
    const rund::kernel::BindingValidation binding =
        BindingPlanCheck(plan, bindings, artifact.metadata,
                         PlanBindingInputMode::StagedOrResident);
    if (!binding.ok) {
      SetMetalLastError(*adapter, binding.reason);
      return false;
    }
    if (!RuntimeWindowsMatchPlan(plan, windows, window_count, bindings)) {
      SetMetalLastError(*adapter, "compute_dispatch_count_mismatch");
      return false;
    }
    std::array<InputWindowPlan, rund::kernel::kMaxComputeBindingCount>
        input_plan_storage{};
    const std::span<InputWindowPlan> input_plans{
        input_plan_storage.data(),
        static_cast<std::size_t>(plan.input_buffer_count)};
    if (!FreezeInputWindowPlans(artifact.metadata, plan.tile_count,
                                input_plans)) {
      SetMetalLastError(*adapter, "compute_binding_mismatch");
      return false;
    }
    std::shared_ptr<void> pipeline =
        MetalPipelineForArtifact(*adapter, artifact);
    if (pipeline == nullptr) {
      return false;
    }

    MetalResidentBindings resident{};
    if (bindings.has_resident_output() &&
        !PrepareResidentBindings(*adapter, plan, bindings, resident)) {
      return false;
    }
    ScopedMetalBuffers params{*adapter};
    MetalRuntimeBuffer &param_buffer = params.add(AcquireMetalBuffer(
        *adapter, plan.param_bytes, MetalBufferUsage::Param));
    if (param_buffer.buffer == nullptr) {
      return false;
    }
    if (bindings.has_resident_output()) {
      if (!UploadMetalBufferUncounted(param_buffer, bindings.param_data,
                                      plan.param_bytes)) {
        return false;
      }
      if (!ExecuteWindows(*adapter, pipeline, plan, windows, window_count,
                          bindings, param_buffer, resident, input_plans)) {
        return false;
      }
    } else if (plan.param_bytes != 0u &&
               !UploadMetalBuffer(*adapter, param_buffer, bindings.param_data,
                                  plan.param_bytes)) {
      return false;
    } else {
      for (rund::kernel::u64 index = 0u; index < window_count; ++index) {
        if (!ExecuteWindow(*adapter, pipeline, plan, windows[index], bindings,
                           &param_buffer, input_plans)) {
          return false;
        }
      }
    }
    SetMetalLastError(*adapter, "ok");
    return true;
  }
}
#else
bool ExecuteMetal(void *, const rund::kernel::ComputePlan &,
                  const rund::kernel::LoweringArtifact &,
                  const rund::kernel::ComputeDispatchWindow *,
                  rund::kernel::u64, const rund::kernel::BindingSet &) {
  return false;
}
#endif

} // namespace rund::node::accel::detail
