#include <accel/check.hpp>
#include <accel/device.hpp>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include "../../window/metal.hpp"
#include "../../window/shape.hpp"
#include "../command/run.hpp"
#include "../range/local.hpp"
#include "../range/resources/lookup.hpp"

#include <string_view>

namespace rund::node::accel::detail {

rund::AccelCheck PrepareMetalWindow(
    const rund::AccelDevice &pick, const rund::kernel::WindowDesc &desc,
    const rund::kernel::WindowPlan &plan, const RangeBinds &bindings,
    const RangePlan &range, const BoundControl *const control,
    std::shared_ptr<void> &resources,
    const MetalKernelImmutablePipelines *const pipelines) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  resources.reset();
  auto *const adapter = static_cast<MetalAdapter *>(pick.backend.context);
  if (!MetalPickOwnsAdapter(pick) || adapter == nullptr) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  if (!WindowShapeOk(desc, plan, bindings) ||
      !WindowRangePlanMatches(plan, range)) {
    SetMetalLastError(*adapter, "compute_window_invalid");
    return rund::AccelCheck{false, "compute_window_invalid"};
  }
  std::optional<MetalRangeBinds> range_bindings;
  const rund::AccelCheck lookup =
      LookupMetalRangeResidentBuffers(pick, bindings, range_bindings);
  if (!lookup.ok || !range_bindings.has_value()) {
    SetMetalLastError(*adapter, lookup.reason);
    return lookup;
  }
  const rund::AccelCheck check = PrepareMetalRange(
      pick, range, *range_bindings, control, resources, pipelines);
  if (!check.ok &&
      std::string_view{check.reason} == "compute_range_aggregate_invalid") {
    SetMetalLastError(*adapter, "compute_window_invalid");
    return rund::AccelCheck{false, "compute_window_invalid"};
  }
  return check;
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)bindings;
  (void)range;
  (void)control;
  (void)resources;
  (void)pipelines;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

rund::AccelCheck EncodeMetalWindow(MetalAdapter &adapter,
                                   const std::shared_ptr<void> &resources,
                                   void *const command_encoder) {
  const rund::AccelCheck check =
      EncodeMetalRange(adapter, resources, command_encoder);
  if (!check.ok &&
      std::string_view{check.reason} == "compute_range_aggregate_invalid") {
    SetMetalLastError(adapter, "compute_window_invalid");
    return rund::AccelCheck{false, "compute_window_invalid"};
  }
  return check;
}

} // namespace rund::node::accel::detail
