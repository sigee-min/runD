#include "local.hpp"
#include "../../api.hpp"

#include <limits>
#include <utility>

namespace rund::node::accel::detail::metal_map_prepare {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

void destroy_resources(void *const raw) {
  auto *const resources = static_cast<MetalMapEncodeResources *>(raw);
  if (resources == nullptr) {
    return;
  }
  if (resources->adapter != nullptr) {
    ReleaseMetalBuffer(*resources->adapter, std::move(resources->param));
    ReleaseMetalBuffer(*resources->adapter, std::move(resources->control_args));
    ReleaseMetalBuffer(*resources->adapter, std::move(resources->control_params));
    ReleaseMetalBuffer(*resources->adapter, std::move(resources->control_status));
  }
  delete resources;
}

} // namespace

rund::AccelCheck prepare_route_resources(
    MetalAdapter &adapter, const rund::AccelDevice &pick,
    const rund::kernel::ComputePlan &plan,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings, const BoundControl &control,
    std::shared_ptr<const MetalMapTemplateResources> prepared,
    std::shared_ptr<void> &resources, const std::uint32_t iterations) {
  auto *const raw = new MetalMapEncodeResources{};
  std::shared_ptr<void> owned{raw, destroy_resources};
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
#endif

} // namespace rund::node::accel::detail::metal_map_prepare
