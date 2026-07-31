#include <accel/check.hpp>
#include <accel/device.hpp>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include "../../stencil/shape.hpp"
#include "../command/run.hpp"
#include "encode/dispatch.hpp"
#include "local.hpp"
#include "pipeline/store.hpp"
#include "resources/pipeline.hpp"

namespace rund::node::accel::detail {

void DestroyMetalStencilEncodeResources(void *const raw) {
  auto *const resources = static_cast<MetalStencilEncodeResources *>(raw);
  delete resources;
}

bool CompileMetalStencilPipeline(MetalAdapter &adapter,
                                 const rund::kernel::StencilOp op,
                                 const rund::kernel::StencilElement element,
                                 const rund::kernel::ComputeDomain domain,
                                 std::shared_ptr<void> &out) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  const std::string key = StencilPipelineKey(op, element, domain);
  if (LookupMetalStencilPipeline(adapter, key, out)) {
    return true;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  if (device == nil) {
    return false;
  }
  const std::uint64_t create_begin = MonotonicNanoseconds();
  if (!CompileMetalStencilPipelineLibrary(adapter, op, element, domain, out)) {
    return false;
  }
  StoreMetalStencilPipeline(adapter, key, out,
                            MonotonicNanoseconds() - create_begin);
  return true;
#else
  (void)adapter;
  (void)op;
  (void)element;
  (void)domain;
  (void)out;
  return false;
#endif
}

rund::AccelCheck PrepareMetalStencil(const rund::AccelDevice &pick,
                                     const rund::kernel::StencilDesc &desc,
                                     const rund::kernel::StencilPlan &plan,
                                     const rund::kernel::ComputeDomain domain,
                                     const StencilBinds &bindings,
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
  if (!StencilShapeOk(desc, plan, bindings)) {
    SetMetalLastError(*adapter, "compute_stencil_invalid");
    return rund::AccelCheck{false, "compute_stencil_invalid"};
  }

  auto *const raw = new MetalStencilEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyMetalStencilEncodeResources};
  raw->adapter = adapter;
  raw->plan = plan;
  rund::AccelCheck check =
      LookupMetalStencilResidentBuffers(pick, bindings, *raw);
  if (!check.ok) {
    SetMetalLastError(*adapter, check.reason);
    return check;
  }
  check = PrepareMetalStencilPipeline(*adapter, plan, domain, *raw);
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
  (void)domain;
  (void)bindings;
  (void)resources;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

rund::AccelCheck EncodeMetalStencil(MetalAdapter &adapter,
                                    const std::shared_ptr<void> &resources,
                                    void *command_encoder) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  MetalStencilCommandState state{};
  const rund::AccelCheck prepared = PrepareMetalStencilCommandState(
      adapter, resources, command_encoder, state);
  if (!prepared.ok) {
    return prepared;
  }
  const StencilParams params{state.stencil->plan.element_count,
                             state.stencil->plan.radius};
  EncodeMetalStencilDispatch(state, params);
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  (void)command_encoder;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

rund::AccelCheck ExecuteMetalStencil(const rund::AccelDevice &pick,
                                     const rund::kernel::StencilDesc &desc,
                                     const rund::kernel::StencilPlan &plan,
                                     const rund::kernel::ComputeDomain domain,
                                     const StencilBinds &bindings) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const adapter = static_cast<MetalAdapter *>(pick.backend.context);
  if (!MetalPickOwnsAdapter(pick) || adapter == nullptr ||
      adapter->queue == nullptr) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  std::shared_ptr<void> resources{};
  const rund::AccelCheck prepare =
      PrepareMetalStencil(pick, desc, plan, domain, bindings, resources);
  if (!prepare.ok) {
    return prepare;
  }

  CommandRun command{};
  const rund::AccelCheck open = OpenCommand(*adapter, command);
  if (!open.ok) {
    return open;
  }
  const rund::AccelCheck encode =
      EncodeMetalStencil(*adapter, resources, (__bridge void *)command.encoder);
  const rund::AccelCheck submit = FinishCommand(*adapter, command, encode);
  if (!submit.ok) {
    return submit;
  }
  return FinishMetalStencil(*adapter, resources);
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)domain;
  (void)bindings;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
