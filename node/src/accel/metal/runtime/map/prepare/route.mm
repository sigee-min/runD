#include "local.hpp"

#include "../admission.hpp"

#include <utility>

namespace rund::node::accel::detail {

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
      !metal_map_prepare::same_bindings(*prepared, bindings)) {
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }
  const rund::AccelCheck valid = ValidateMetalMapPrepare(
      *adapter, plan, artifact, windows, window_count, bindings);
  if (!valid.ok) {
    return valid;
  }
  return metal_map_prepare::prepare_route_resources(
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
  (void)prepared;
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
  return metal_map_prepare::prepare_route_resources(
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

} // namespace rund::node::accel::detail
