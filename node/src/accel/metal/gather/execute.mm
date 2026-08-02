#include <accel/check.hpp>
#include <accel/device.hpp>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include "../../gather/shape.hpp"
#include "../command/run.hpp"
#include "encode/dispatch.hpp"
#include "local.hpp"
#include "pipeline/store.hpp"
#include "resources/pipeline.hpp"
#include "../pipeline/template.hpp"

#include <utility>

namespace rund::node::accel::detail {

void DestroyMetalGatherEncodeResources(void *const raw) {
  auto *const resources = static_cast<MetalGatherEncodeResources *>(raw);
  if (resources == nullptr) {
    return;
  }
  if (resources->adapter != nullptr) {
    ReleaseMetalBuffer(*resources->adapter, std::move(resources->status));
    ReleaseMetalBuffer(*resources->adapter, std::move(resources->indirect));
  }
  delete resources;
}

bool CompileMetalGatherPipelines(MetalAdapter &adapter,
                                 const rund::kernel::GatherElement element,
                                 std::shared_ptr<void> &control,
                                 std::shared_ptr<void> &gather) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  const std::string key = GatherPipelineKey(element);
  control = LookupMetalNamedPipeline(adapter, key + ".control");
  gather = LookupMetalNamedPipeline(adapter, key + ".copy");
  if (control != nullptr && gather != nullptr) {
    return true;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  if (device == nil) {
    return false;
  }
  const std::uint64_t create_begin = MonotonicNanoseconds();
  if (!CompileMetalGatherPipelineLibrary(adapter, element, control, gather)) {
    return false;
  }
  const std::uint64_t elapsed = MonotonicNanoseconds() - create_begin;
  StoreMetalNamedPipeline(adapter, key + ".control", control, elapsed);
  StoreMetalNamedPipeline(adapter, key + ".copy", gather, 0u);
  return true;
#else
  (void)adapter;
  (void)element;
  (void)control;
  (void)gather;
  return false;
#endif
}

rund::AccelCheck PrepareMetalGather(const rund::AccelDevice &pick,
                                    const rund::kernel::GatherDesc &desc,
                                    const rund::kernel::GatherPlan &plan,
                                    const GatherBinds &bindings,
                                    std::shared_ptr<void> &resources,
                                    const MetalKernelImmutablePipelines *const
                                        pipelines) {
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
  if (!GatherShapeOk(desc, plan, bindings)) {
    SetMetalLastError(*adapter, "compute_gather_invalid");
    return rund::AccelCheck{false, "compute_gather_invalid"};
  }

  auto *const raw = new MetalGatherEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyMetalGatherEncodeResources};
  raw->adapter = adapter;
  raw->plan = plan;
  rund::AccelCheck check =
      LookupMetalGatherResidentBuffers(pick, bindings, *raw);
  if (!check.ok) {
    SetMetalLastError(*adapter, check.reason);
    return check;
  }
  check = PrepareMetalGatherStatusBuffer(*adapter, plan, *raw);
  if (check.ok) {
    if (pipelines != nullptr && pipelines->ready(2u)) {
      raw->control_pipeline = pipelines->stages[0u];
      raw->gather_pipeline = pipelines->stages[1u];
    } else if (pipelines != nullptr) {
      check = {false, "accel_metal_pipeline_unavailable"};
    } else {
      check = PrepareMetalGatherPipeline(*adapter, plan, *raw);
    }
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
  (void)pipelines;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

rund::AccelCheck EncodeMetalGather(MetalAdapter &adapter,
                                   const std::shared_ptr<void> &resources,
                                   void *command_encoder) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  MetalGatherCommandState state{};
  const rund::AccelCheck prepared = PrepareMetalGatherCommandState(
      adapter, resources, command_encoder, state);
  if (!prepared.ok) {
    return prepared;
  }
  const GatherParams params{
      state.gather->plan.element_count, state.gather->plan.source_count,
      static_cast<rund::kernel::u32>(state.gather->plan.count_source), 0u};
  EncodeMetalGatherDispatch(state, params);
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  (void)command_encoder;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

rund::AccelCheck ExecuteMetalGather(const rund::AccelDevice &pick,
                                    const rund::kernel::GatherDesc &desc,
                                    const rund::kernel::GatherPlan &plan,
                                    const GatherBinds &bindings) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const adapter = static_cast<MetalAdapter *>(pick.backend.context);
  if (!MetalPickOwnsAdapter(pick) || adapter == nullptr ||
      adapter->queue == nullptr) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  std::shared_ptr<void> resources{};
  const rund::AccelCheck prepare =
      PrepareMetalGather(pick, desc, plan, bindings, resources);
  if (!prepare.ok) {
    return prepare;
  }

  CommandRun command{};
  const rund::AccelCheck open = OpenCommand(*adapter, command);
  if (!open.ok) {
    return open;
  }
  const rund::AccelCheck encode =
      EncodeMetalGather(*adapter, resources, (__bridge void *)command.encoder);
  const rund::AccelCheck submit = FinishCommand(*adapter, command, encode);
  if (!submit.ok) {
    return submit;
  }
  return FinishMetalGather(*adapter, resources);
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)bindings;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
