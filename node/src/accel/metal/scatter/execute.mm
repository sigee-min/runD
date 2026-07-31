#include <accel/check.hpp>
#include <accel/device.hpp>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include "../../scatter/shape.hpp"
#include "../command/run.hpp"
#include "encode/dispatch.hpp"
#include "local.hpp"
#include "pipeline/store.hpp"
#include "resources/pipeline.hpp"

#include <utility>

namespace rund::node::accel::detail {

void DestroyMetalScatterEncodeResources(void *const raw) {
  auto *const resources = static_cast<MetalScatterEncodeResources *>(raw);
  if (resources == nullptr) {
    return;
  }
  if (resources->adapter != nullptr) {
    ReleaseMetalBuffer(*resources->adapter, std::move(resources->status));
  }
  delete resources;
}

bool CompileMetalScatterPipeline(MetalAdapter &adapter,
                                 const rund::kernel::ScatterElement element,
                                 std::shared_ptr<void> &out) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  const std::string key = ScatterPipelineKey(element);
  if (LookupMetalScatterPipeline(adapter, key, out)) {
    return true;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  if (device == nil) {
    return false;
  }
  const std::uint64_t create_begin = MonotonicNanoseconds();
  if (!CompileMetalScatterPipelineLibrary(adapter, element, out)) {
    return false;
  }
  StoreMetalScatterPipeline(adapter, key, out,
                            MonotonicNanoseconds() - create_begin);
  return true;
#else
  (void)adapter;
  (void)element;
  (void)out;
  return false;
#endif
}

rund::AccelCheck PrepareMetalScatter(const rund::AccelDevice &pick,
                                     const rund::kernel::ScatterDesc &desc,
                                     const rund::kernel::ScatterPlan &plan,
                                     const ScatterBinds &bindings,
                                     std::shared_ptr<void> &resources) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  resources.reset();
  if (!MetalPickOwnsAdapter(pick)) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  auto *const adapter = static_cast<MetalAdapter *>(pick.backend.context);
  if (adapter == nullptr || adapter->device == nullptr) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  SetMetalLastError(*adapter, "ok");
  if (!ScatterShapeOk(desc, plan, bindings)) {
    SetMetalLastError(*adapter, "compute_scatter_invalid");
    return rund::AccelCheck{false, "compute_scatter_invalid"};
  }

  auto *const raw = new MetalScatterEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyMetalScatterEncodeResources};
  raw->adapter = adapter;
  raw->plan = plan;
  rund::AccelCheck check =
      LookupMetalScatterResidentBuffers(pick, bindings, *raw);
  if (check.ok) {
    check = PrepareMetalScatterStatusBuffer(*adapter, plan, *raw);
  }
  if (check.ok) {
    check = PrepareMetalScatterPipeline(*adapter, plan, *raw);
  }
  if (!check.ok) {
    SetMetalLastError(*adapter, check.reason);
    return check;
  }
  resources = std::move(owned);
  return rund::AccelCheck{true, "ok"};
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)bindings;
  (void)resources;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

rund::AccelCheck EncodeMetalScatter(MetalAdapter &adapter,
                                    const std::shared_ptr<void> &resources,
                                    void *command_encoder) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  MetalScatterCommandState state{};
  const rund::AccelCheck prepared = PrepareMetalScatterCommandState(
      adapter, resources, command_encoder, state);
  if (!prepared.ok) {
    return prepared;
  }
  const ScatterParams params{state.scatter->plan.element_count,
                             state.scatter->plan.output_count};
  EncodeMetalScatterDispatch(state, params);
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  (void)command_encoder;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

rund::AccelCheck ExecuteMetalScatter(const rund::AccelDevice &pick,
                                     const rund::kernel::ScatterDesc &desc,
                                     const rund::kernel::ScatterPlan &plan,
                                     const ScatterBinds &bindings) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const adapter = static_cast<MetalAdapter *>(pick.backend.context);
  if (!MetalPickOwnsAdapter(pick) || adapter == nullptr ||
      adapter->queue == nullptr) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  std::shared_ptr<void> resources{};
  const rund::AccelCheck prepare =
      PrepareMetalScatter(pick, desc, plan, bindings, resources);
  if (!prepare.ok) {
    return prepare;
  }

  CommandRun command{};
  const rund::AccelCheck open = OpenCommand(*adapter, command);
  if (!open.ok) {
    return open;
  }
  const rund::AccelCheck encode =
      EncodeMetalScatter(*adapter, resources, (__bridge void *)command.encoder);
  const rund::AccelCheck submit = FinishCommand(*adapter, command, encode);
  if (!submit.ok) {
    return submit;
  }
  return FinishMetalScatter(*adapter, resources);
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)bindings;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
