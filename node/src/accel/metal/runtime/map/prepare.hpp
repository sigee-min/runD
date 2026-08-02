#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../../kernel/backend/template_plan.hpp"
#include "../../../kernel/step/map/stride.hpp"
#include "../../pipeline/guard.hpp"
#include "admission.hpp"
#include "api.hpp"
#include "control.hpp"
#include "lifetime.hpp"

#include <algorithm>
#include <limits>
#include <utility>

namespace rund::node::accel::detail {

namespace {

[[nodiscard]] bool SameMetalMapTemplateBindings(
    const MetalMapTemplateResources &prepared,
    const rund::kernel::BindingSet &bindings) noexcept {
  if (prepared.input_strides.size() != bindings.resident_inputs.count ||
      prepared.output_strides.size() != bindings.resident_outputs.count) {
    return false;
  }
  for (std::size_t index = 0u; index < prepared.input_strides.size(); ++index) {
    const auto *const ref = bindings.resident_inputs.ref(index);
    if (ref == nullptr || ref->stride_bytes != prepared.input_strides[index]) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < prepared.output_strides.size();
       ++index) {
    const auto *const ref = bindings.resident_outputs.ref(index);
    if (ref == nullptr || ref->stride_bytes != prepared.output_strides[index]) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] rund::AccelCheck PrepareMetalMapRouteResources(
    MetalAdapter &adapter, const rund::AccelDevice &pick,
    const rund::kernel::ComputePlan &plan,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings, const BoundControl &control,
    std::shared_ptr<const MetalMapTemplateResources> prepared,
    std::shared_ptr<void> &resources, const std::uint32_t iterations) {
  auto *const raw = new MetalMapEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyMetalMapEncodeResources};
  raw->adapter = &adapter;
  raw->prepared = std::move(prepared);
  raw->bindings = bindings;
  raw->iterations = iterations;
  if (window_count > std::numeric_limits<std::size_t>::max()) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  raw->windows.reserve(static_cast<std::size_t>(window_count));
  if (raw->windows.capacity() != window_count) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  raw->windows.assign(windows, windows + window_count);
  if (raw->windows.capacity() != window_count) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  if (!PrepareResidentBindings(adapter, raw->prepared->plan, raw->bindings,
                               raw->resident)) {
    SetMetalLastError(adapter, "compute_binding_mismatch");
    return rund::AccelCheck{false, "compute_binding_mismatch"};
  }
  raw->param =
      AcquireMetalBuffer(adapter, plan.param_bytes, MetalBufferUsage::Param);
  if (raw->param.buffer == nullptr ||
      !UploadMetalBufferUncounted(raw->param, bindings.param_data,
                                  plan.param_bytes) ||
      !PrepareMetalMapControl(pick, control, raw->windows, *raw)) {
    SetMetalLastError(adapter, "accel_metal_command_unavailable");
    return rund::AccelCheck{false, "accel_metal_command_unavailable"};
  }
  resources = std::move(owned);
  return rund::AccelCheck{true, "ok"};
}

} // namespace

bool MetalMapTemplateMatches(
    const MetalMapTemplateResources &prepared, const MetalAdapter &adapter,
    const rund::kernel::ComputePlan &plan,
    const rund::kernel::BindingSet &bindings) noexcept {
  return prepared.adapter == &adapter &&
         backend_template_plan::same_plan(prepared.plan, plan) &&
         SameMetalMapTemplateBindings(prepared, bindings);
}

[[nodiscard]] inline rund::AccelCheck PrepareMetalMapTemplateImpl(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    rund::kernel::LoweringArtifact *const owned_artifact,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings, const BoundControl &control,
    std::shared_ptr<const MetalMapTemplateResources> &prepared) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  prepared.reset();
  auto *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr || adapter->device == nullptr) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  SetMetalLastError(*adapter, "ok");
  const rund::AccelCheck valid = ValidateMetalMapPrepare(
      *adapter, plan, artifact, windows, window_count, bindings);
  if (!valid.ok) {
    return valid;
  }

  auto owned = std::make_shared<MetalMapTemplateResources>();
  MetalMapTemplateResources *const raw = owned.get();
  raw->adapter = adapter;
  raw->plan = plan;
  raw->input_plans.reserve(static_cast<std::size_t>(plan.input_buffer_count));
  if (raw->input_plans.capacity() != plan.input_buffer_count) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  raw->input_plans.resize(static_cast<std::size_t>(plan.input_buffer_count));
  if (!FreezeInputWindowPlans(artifact.metadata, plan.tile_count,
                              raw->input_plans)) {
    SetMetalLastError(*adapter, "compute_binding_mismatch");
    return rund::AccelCheck{false, "compute_binding_mismatch"};
  }
  raw->input_strides.reserve(
      static_cast<std::size_t>(bindings.resident_inputs.count));
  if (raw->input_strides.capacity() != bindings.resident_inputs.count) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  for (std::uint64_t index = 0u; index < bindings.resident_inputs.count;
       ++index) {
    const auto *const ref = bindings.resident_inputs.ref(index);
    if (ref == nullptr) {
      return rund::AccelCheck{false, "compute_binding_mismatch"};
    }
    raw->input_strides.push_back(ref->stride_bytes);
  }
  raw->output_strides.reserve(
      static_cast<std::size_t>(bindings.resident_outputs.count));
  if (raw->output_strides.capacity() != bindings.resident_outputs.count) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  for (std::uint64_t index = 0u; index < bindings.resident_outputs.count;
       ++index) {
    const auto *const ref = bindings.resident_outputs.ref(index);
    if (ref == nullptr) {
      return rund::AccelCheck{false, "compute_binding_mismatch"};
    }
    raw->output_strides.push_back(ref->stride_bytes);
  }
  const std::uint64_t check_count = MetalMapUniqueCheckCount(artifact);
  if (check_count > std::numeric_limits<std::size_t>::max()) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  raw->checks.reserve(static_cast<std::size_t>(check_count));
  if (raw->checks.capacity() != check_count) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  for (const rund::kernel::ReadRoute route : artifact.metadata.read_routes) {
    const auto found = std::find_if(raw->checks.begin(), raw->checks.end(),
                                    [&](const MetalMapCheck check) {
                                      return check.binding == route.index;
                                    });
    if (found == raw->checks.end()) {
      raw->checks.push_back(MetalMapCheck{route.index, route.count});
    } else {
      found->limit =
          std::min(found->limit, static_cast<std::uint64_t>(route.count));
    }
  }
  if (raw->checks.size() != check_count ||
      raw->checks.capacity() != check_count) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  const bool controlled_source = control.active() || !raw->checks.empty();
  std::uint64_t specialized_upper = 0u;
  std::uint64_t controlled_upper = 0u;
  std::uint64_t final_upper = 0u;
  if (!MapSpecializedSourceUpperBytes(artifact, plan, specialized_upper) ||
      (controlled_source && !MetalControlledMapSourceUpperBytes(
                                plan, specialized_upper, controlled_upper)) ||
      !PipelinePrivateMetalSourceUpperBytes(
          controlled_source ? controlled_upper : specialized_upper, 1u,
          IsPipelinePrivatePreparation(CurrentKernelPreparationMode()),
          final_upper)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  rund::kernel::LoweringArtifact specialized =
      owned_artifact == nullptr
          ? SpecializeMap(artifact, plan, bindings, 1u, final_upper)
          : SpecializeMapInPlace(std::move(*owned_artifact), plan, bindings, 1u,
                                 final_upper);
  rund::kernel::LoweringArtifact controlled =
      controlled_source
          ? MetalControlledMapArtifact(std::move(specialized), plan)
          : std::move(specialized);
  if (!controlled.ok) {
    SetMetalLastError(*adapter, controlled.reason);
    return rund::AccelCheck{false, controlled.reason};
  }
  raw->pipeline = MetalPipelineForArtifact(*adapter, std::move(controlled));
  if (raw->pipeline == nullptr) {
    SetMetalLastError(*adapter, "accel_metal_command_unavailable");
    return rund::AccelCheck{false, "accel_metal_command_unavailable"};
  }
  if (controlled_source) {
    raw->control_pipeline = MetalMapControlPipeline(*adapter);
  }
  if (!raw->checks.empty()) {
    raw->check_pipeline = MetalPipelineForArtifact(
        *adapter, MetalMapCheckArtifact(*raw, bindings));
  }
  if ((controlled_source && raw->control_pipeline == nullptr) ||
      (!raw->checks.empty() && raw->check_pipeline == nullptr)) {
    SetMetalLastError(*adapter, "accel_metal_command_unavailable");
    return rund::AccelCheck{false, "accel_metal_command_unavailable"};
  }
  prepared = std::move(owned);
  return rund::AccelCheck{true, "ok"};
#else
  (void)pick;
  (void)plan;
  (void)artifact;
  (void)owned_artifact;
  (void)windows;
  (void)window_count;
  (void)bindings;
  (void)control;
  (void)prepared;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

rund::AccelCheck PrepareMetalMapTemplate(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings, const BoundControl &control,
    std::shared_ptr<const MetalMapTemplateResources> &prepared) {
  return PrepareMetalMapTemplateImpl(pick, plan, artifact, nullptr, windows,
                                     window_count, bindings, control, prepared);
}

rund::AccelCheck PrepareMetalMapOwnedTemplate(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    rund::kernel::LoweringArtifact &&artifact,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings, const BoundControl &control,
    std::shared_ptr<const MetalMapTemplateResources> &prepared) {
  return PrepareMetalMapTemplateImpl(pick, plan, artifact, &artifact, windows,
                                     window_count, bindings, control, prepared);
}

rund::AccelCheck PrepareMetalMapRoute(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings, const BoundControl &control,
    std::shared_ptr<const MetalMapTemplateResources> prepared,
    std::shared_ptr<void> &resources, const std::uint32_t iterations) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  resources.reset();
  if (iterations == 0u || (iterations != 1u && control.active())) {
    return rund::AccelCheck{false, "compute_pipeline_invalid"};
  }
  auto *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr || adapter->device == nullptr || prepared == nullptr ||
      prepared->adapter != adapter ||
      prepared->plan.op_hash_hi != plan.op_hash_hi ||
      prepared->plan.op_hash_lo != plan.op_hash_lo ||
      prepared->plan.dispatch_count != plan.dispatch_count ||
      !SameMetalMapTemplateBindings(*prepared, bindings)) {
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }
  const rund::AccelCheck valid = ValidateMetalMapPrepare(
      *adapter, plan, artifact, windows, window_count, bindings);
  if (!valid.ok) {
    return valid;
  }
  return PrepareMetalMapRouteResources(
      *adapter, pick, plan, windows, window_count, bindings, control,
      std::move(prepared), resources, iterations);
#else
  (void)pick;
  (void)plan;
  (void)artifact;
  (void)windows;
  (void)window_count;
  (void)bindings;
  (void)control;
  (void)resources;
  (void)iterations;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

rund::AccelCheck PrepareMetalMapProvedRoute(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings, const BoundControl &control,
    std::shared_ptr<const MetalMapTemplateResources> prepared,
    std::shared_ptr<void> &resources, const std::uint32_t iterations) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  resources.reset();
  if (iterations == 0u || (iterations != 1u && control.active())) {
    return rund::AccelCheck{false, "compute_pipeline_invalid"};
  }
  auto *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr || adapter->device == nullptr || prepared == nullptr ||
      !MetalMapTemplateMatches(*prepared, *adapter, plan, bindings)) {
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }
  // Source/artifact admission belongs to the common recurrence proof and the
  // immutable template miss path. Only route-varying runtime state remains.
  if (!RuntimeWindowsMatchPlan(plan, windows, window_count, bindings) ||
      !bindings.has_resident_output()) {
    SetMetalLastError(*adapter, "compute_dispatch_count_mismatch");
    return rund::AccelCheck{false, "compute_dispatch_count_mismatch"};
  }
  return PrepareMetalMapRouteResources(
      *adapter, pick, plan, windows, window_count, bindings, control,
      std::move(prepared), resources, iterations);
#else
  (void)pick;
  (void)plan;
  (void)windows;
  (void)window_count;
  (void)bindings;
  (void)control;
  (void)prepared;
  (void)resources;
  (void)iterations;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

rund::AccelCheck PrepareMetalMap(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings, const BoundControl &control,
    std::shared_ptr<void> &resources, const std::uint32_t iterations) {
  std::shared_ptr<const MetalMapTemplateResources> prepared;
  const rund::AccelCheck template_ready = PrepareMetalMapTemplate(
      pick, plan, artifact, windows, window_count, bindings, control, prepared);
  return template_ready.ok
             ? PrepareMetalMapRoute(pick, plan, artifact, windows, window_count,
                                    bindings, control, std::move(prepared),
                                    resources, iterations)
             : template_ready;
}

} // namespace rund::node::accel::detail
