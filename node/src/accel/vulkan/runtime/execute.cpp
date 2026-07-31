#include "local.hpp"
#include <rund/counter.hpp>

#include <kernel/program/compute/lowering/artifact/admission.hpp>

#include <array>
#include <mutex>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
bool ExecuteVulkan(void *const context, const rund::kernel::ComputePlan &plan,
                   const rund::kernel::LoweringArtifact &artifact,
                   const rund::kernel::ComputeDispatchWindow *const windows,
                   const rund::kernel::u64 window_count,
                   const rund::kernel::BindingSet &bindings) {
  auto *const adapter = static_cast<VulkanAdapter *>(context);
  if (adapter == nullptr || adapter->device == VK_NULL_HANDLE ||
      adapter->compute_queue == VK_NULL_HANDLE) {
    return false;
  }
  std::lock_guard<std::mutex> lock{adapter->mutex};
  SetVulkanLastError(*adapter, "ok");
  if (!ComputePlanHeaderValid(plan, rund::kernel::ComputeApi::Vulkan)) {
    SetVulkanLastError(*adapter, "compute_plan_invalid");
    return false;
  }
  if (!FrozenCapsAdmitPlan(adapter->caps, plan)) {
    SetVulkanLastError(*adapter, "compute_backend_mismatch");
    return false;
  }
  const rund::kernel::compute_lowering_detail::ArtifactAdmission admission =
      rund::kernel::compute_lowering_detail::AdmitArtifact(plan, artifact);
  if (!admission.ok) {
    SetVulkanLastError(*adapter,
                       artifact.kind ==
                               rund::kernel::LoweringArtifactKind::VulkanSource
                           ? "compute_artifact_mismatch"
                           : "compute_artifact_non_executable");
    return false;
  }
  const rund::kernel::BindingValidation binding =
      BindingPlanCheck(plan, bindings, artifact.metadata,
                       PlanBindingInputMode::StagedOrResident);
  if (!binding.ok) {
    SetVulkanLastError(*adapter, binding.reason);
    return false;
  }
  if (!RuntimeWindowsMatchPlan(plan, windows, window_count, bindings)) {
    SetVulkanLastError(*adapter, "compute_dispatch_count_mismatch");
    return false;
  }
  std::array<InputWindowPlan, rund::kernel::kMaxComputeBindingCount>
      input_plan_storage{};
  const std::span<InputWindowPlan> input_plans{
      input_plan_storage.data(),
      static_cast<std::size_t>(plan.input_buffer_count)};
  if (!FreezeInputWindowPlans(artifact.metadata, plan.tile_count,
                              input_plans)) {
    SetVulkanLastError(*adapter, "compute_binding_mismatch");
    return false;
  }
  VulkanCachedPipeline *const pipeline =
      AcquireVulkanCachedPipeline(*adapter, plan, artifact);
  if (pipeline == nullptr) {
    return false;
  }
  VulkanResidentBindings resident_bindings{};
  VulkanResidentBindings *resident_ptr = nullptr;
  if (bindings.has_resident_output()) {
    if (!PrepareVulkanResidentBindings(*adapter, plan, bindings,
                                       resident_bindings)) {
      SetVulkanLastError(*adapter, resident_bindings.reason);
      return false;
    }
    resident_ptr = &resident_bindings;
  }
  ScopedBuffer param_buffer{};
  if (!MakeHostBuffer(*adapter, bindings.param_data, plan.param_bytes,
                      param_buffer)) {
    return false;
  }
  if (!bindings.has_resident_output()) {
    ::rund::detail::counter::Accumulate(adapter->host_to_device_bytes,
                                        plan.param_bytes);
  }
  for (rund::kernel::u64 index = 0u; index < window_count; ++index) {
    if (!ExecuteWindow(*adapter, *pipeline, plan, windows[index], bindings,
                       input_plans, param_buffer, resident_ptr)) {
      return false;
    }
  }
  SetVulkanLastError(*adapter, "ok");
  return true;
}
#endif

} // namespace rund::node::accel::detail
