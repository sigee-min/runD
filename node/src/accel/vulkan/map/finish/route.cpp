#include "../../adapter/error.hpp"
#include "../../adapter/access.hpp"

#include "../admission.hpp"
#include "../api.hpp"
#include "../local.hpp"
#include "../resources/admission.hpp"

#include "../../collective/pipeline.hpp"

#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace {

[[nodiscard]] bool PrepareStandaloneVulkanMapDescriptorDemand(
    VulkanAdapter &adapter, const VulkanMapTemplateResources &prepared,
    std::shared_ptr<VulkanMapDescriptorArena> &descriptors) {
  if (prepared.pipeline == nullptr || prepared.plan.dispatch_count == 0u ||
      !PrepareVulkanMapDescriptorArena(adapter, *prepared.pipeline,
                                       prepared.plan.dispatch_count,
                                       descriptors)) {
    return false;
  }
  if (prepared.control_pipeline != nullptr &&
      !ReserveVulkanCollectiveDescriptorDemand(
          adapter, *prepared.control_pipeline,
          prepared.control_pipeline->descriptor_count, 1u)) {
    return false;
  }
  if (prepared.check_pipeline != nullptr &&
      !ReserveVulkanCollectiveDescriptorDemand(
          adapter, *prepared.check_pipeline,
          prepared.check_pipeline->descriptor_count, 1u)) {
    return false;
  }
  return true;
}

[[nodiscard]] rund::AccelCheck PrepareVulkanMapRouteResources(
    VulkanAdapter &adapter, const rund::AccelDevice &pick,
    const rund::kernel::ComputePlan &plan,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings, const BoundControl &control,
    const bool history_recurrence,
    std::shared_ptr<const VulkanMapTemplateResources> prepared,
    std::shared_ptr<VulkanMapDescriptorArena> descriptors,
    std::shared_ptr<void> &resources, const rund::kernel::u32 iterations) {
  auto *const raw = new VulkanMapEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyVulkanMapEncodeResources};
  raw->adapter = &adapter;
  raw->prepared = std::move(prepared);
  raw->iterations = iterations;
  raw->history_recurrence = history_recurrence;
  raw->bindings = bindings;
  raw->windows.assign(windows, windows + window_count);
  const char *history_reason = "ok";
  if (!ValidateVulkanMapHistoryOutputs(*raw, history_reason)) {
    SetVulkanLastError(adapter, history_reason);
    return rund::AccelCheck{false, history_reason};
  }
  if (!PrepareVulkanResidentBindings(adapter, raw->prepared->plan,
                                     raw->bindings, raw->resident)) {
    SetVulkanLastError(adapter, raw->resident.reason);
    return rund::AccelCheck{false, raw->resident.reason};
  }
  if (!MakeVulkanMapHostBuffer(adapter, bindings.param_data, plan.param_bytes,
                               raw->param)) {
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }
  if (!PrepareVulkanMapControl(pick, control, raw->windows, *raw)) {
    const char *const reason = VulkanLastError(&adapter);
    return rund::AccelCheck{false, reason == nullptr || reason[0] == '\0'
                                       ? "compute_plan_invalid"
                                       : reason};
  }
  if (!AcquireVulkanMapDescriptorSets(descriptors, window_count,
                                      raw->descriptor_sets)) {
    return rund::AccelCheck{false, "accel_vulkan_descriptor_unavailable"};
  }
  resources = std::move(owned);
  return rund::AccelCheck{true, "ok"};
}

} // namespace

#endif

rund::AccelCheck PrepareVulkanMapRoute(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings, const BoundControl &control,
    std::shared_ptr<const VulkanMapTemplateResources> prepared,
    std::shared_ptr<VulkanMapDescriptorArena> descriptors,
    std::shared_ptr<void> &resources, const rund::kernel::u32 iterations) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  resources.reset();
  const bool history_recurrence =
      artifact.key.variant ==
      rund::kernel::LoweringArtifactVariant::HistoryRecurrence;
  if (iterations == 0u || (history_recurrence && iterations < 2u)) {
    return rund::AccelCheck{false,
                            "compute_pipeline_recurrence_history_invalid"};
  }
  auto *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr || prepared == nullptr ||
      !VulkanMapTemplateMatches(*prepared, *adapter, plan, bindings)) {
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }
  const rund::AccelCheck valid = ValidateVulkanMapPrepare(
      *adapter, plan, artifact, windows, window_count, bindings);
  if (!valid.ok) {
    return valid;
  }
  return PrepareVulkanMapRouteResources(
      *adapter, pick, plan, windows, window_count, bindings, control,
      history_recurrence, std::move(prepared), std::move(descriptors),
      resources, iterations);
#else
  (void)pick;
  (void)plan;
  (void)artifact;
  (void)windows;
  (void)window_count;
  (void)bindings;
  (void)control;
  (void)prepared;
  (void)descriptors;
  (void)resources;
  (void)iterations;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck PrepareVulkanMapProvedRoute(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings, const BoundControl &control,
    const bool history_recurrence,
    std::shared_ptr<const VulkanMapTemplateResources> prepared,
    std::shared_ptr<VulkanMapDescriptorArena> descriptors,
    std::shared_ptr<void> &resources, const rund::kernel::u32 iterations) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  resources.reset();
  if (iterations == 0u || (history_recurrence && iterations < 2u) ||
      (iterations != 1u && control.active())) {
    return rund::AccelCheck{false,
                            history_recurrence
                                ? "compute_pipeline_recurrence_history_invalid"
                                : "compute_pipeline_invalid"};
  }
  auto *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr || prepared == nullptr || descriptors == nullptr ||
      descriptors->adapter != adapter ||
      !VulkanMapTemplateMatches(*prepared, *adapter, plan, bindings)) {
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }
  // Artifact/source admission belongs to common recurrence proof and the
  // immutable template miss path. Only route-varying windows and bindings
  // remain here; rebuilding an artifact would restore the removed layer.
  if (!RuntimeWindowsMatchPlan(plan, windows, window_count, bindings) ||
      !bindings.has_resident_output()) {
    SetVulkanLastError(*adapter, "compute_dispatch_count_mismatch");
    return rund::AccelCheck{false, "compute_dispatch_count_mismatch"};
  }
  return PrepareVulkanMapRouteResources(
      *adapter, pick, plan, windows, window_count, bindings, control,
      history_recurrence, std::move(prepared), std::move(descriptors),
      resources, iterations);
#else
  (void)pick;
  (void)plan;
  (void)windows;
  (void)window_count;
  (void)bindings;
  (void)control;
  (void)history_recurrence;
  (void)prepared;
  (void)descriptors;
  (void)resources;
  (void)iterations;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck PrepareVulkanMap(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings, const BoundControl &control,
    std::shared_ptr<void> &resources, const rund::kernel::u32 iterations) {
  std::shared_ptr<const VulkanMapTemplateResources> prepared;
  const rund::AccelCheck template_ready = PrepareVulkanMapTemplate(
      pick, plan, artifact, windows, window_count, bindings, control, prepared);
  if (!template_ready.ok) {
    return template_ready;
  }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const adapter = CheckedVulkanAdapter(pick);
  std::shared_ptr<VulkanMapDescriptorArena> descriptors;
  if (adapter == nullptr || prepared == nullptr ||
      !PrepareStandaloneVulkanMapDescriptorDemand(*adapter, *prepared,
                                                  descriptors)) {
    return rund::AccelCheck{false, adapter == nullptr
                                       ? "accel_vulkan_unavailable"
                                       : VulkanLastError(adapter)};
  }
  return PrepareVulkanMapRoute(pick, plan, artifact, windows, window_count,
                               bindings, control, std::move(prepared),
                               std::move(descriptors), resources, iterations);
#else
  (void)resources;
  (void)iterations;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
