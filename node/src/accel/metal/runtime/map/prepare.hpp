#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../../kernel/step/map/stride.hpp"
#include "admission.hpp"
#include "control.hpp"
#include "lifetime.hpp"

#include <algorithm>

namespace rund::node::accel::detail {

rund::AccelCheck PrepareMetalMap(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::ComputeDispatchWindow *windows,
    rund::kernel::u64 window_count, const rund::kernel::BindingSet &bindings,
    const BoundControl &control, std::shared_ptr<void> &resources,
    const std::uint32_t iterations) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  resources.reset();
  if (iterations == 0u || (iterations != 1u && control.active())) {
    return rund::AccelCheck{false, "compute_pipeline_invalid"};
  }
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

  auto *const raw = new MetalMapEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyMetalMapEncodeResources};
  raw->adapter = adapter;
  raw->plan = plan;
  raw->bindings = bindings;
  raw->read_routes = artifact.metadata.read_routes;
  for (const rund::kernel::ReadRoute route : raw->read_routes) {
    const auto found = std::find_if(
        raw->checks.begin(), raw->checks.end(),
        [&](const MetalMapCheck check) { return check.binding == route.index; });
    if (found == raw->checks.end()) {
      raw->checks.push_back(MetalMapCheck{route.index, route.count});
    } else {
      found->limit =
          std::min(found->limit, static_cast<std::uint64_t>(route.count));
    }
  }
  raw->iterations = iterations;
  raw->windows.assign(windows, windows + window_count);
  if (!PrepareResidentBindings(*adapter, raw->plan, raw->bindings,
                               raw->resident)) {
    SetMetalLastError(*adapter, "compute_binding_mismatch");
    return rund::AccelCheck{false, "compute_binding_mismatch"};
  }
  const rund::kernel::LoweringArtifact specialized =
      SpecializeMap(artifact, plan, bindings);
  const rund::kernel::LoweringArtifact controlled =
      (control.active() || !raw->checks.empty())
          ? MetalControlledMapArtifact(specialized, plan)
          : specialized;
  if (!controlled.ok) {
    SetMetalLastError(*adapter, controlled.reason);
    return rund::AccelCheck{false, controlled.reason};
  }
  raw->pipeline = MetalPipelineForArtifact(*adapter, controlled);
  raw->param =
      AcquireMetalBuffer(*adapter, plan.param_bytes, MetalBufferUsage::Param);
  if (raw->pipeline == nullptr || raw->param.buffer == nullptr ||
      !UploadMetalBufferUncounted(raw->param, bindings.param_data,
                                  plan.param_bytes) ||
      !PrepareMetalMapControl(pick, control, raw->windows, *raw)) {
    SetMetalLastError(*adapter, "accel_metal_command_unavailable");
    return rund::AccelCheck{false, "accel_metal_command_unavailable"};
  }
  resources = std::move(owned);
  return rund::AccelCheck{true, "ok"};
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

} // namespace rund::node::accel::detail
