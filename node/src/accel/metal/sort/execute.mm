#include <accel/check.hpp>
#include <accel/device.hpp>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include "../../domain.hpp"
#include "../../sort/shape.hpp"
#include "../command/run.hpp"
#include "encode/pass.hpp"
#include "local.hpp"
#include "pipeline/store.hpp"
#include "resources/buffers.hpp"
#include "resources/lookup.hpp"
#include "resources/pipeline.hpp"

#include <utility>

namespace rund::node::accel::detail {

void DestroyMetalSortEncodeResources(void *const raw) {
  auto *const resources = static_cast<MetalSortEncodeResources *>(raw);
  if (resources == nullptr) {
    return;
  }
  if (resources->adapter != nullptr) {
    MetalAdapter &adapter = *resources->adapter;
    ReleaseMetalBuffer(adapter, std::move(resources->temp_keys));
    ReleaseMetalBuffer(adapter, std::move(resources->temp_values));
    ReleaseMetalBuffer(adapter, std::move(resources->block_counts));
    ReleaseMetalBuffer(adapter, std::move(resources->block_offsets));
    ReleaseMetalBuffer(adapter, std::move(resources->bucket_offsets));
    ReleaseMetalBuffer(adapter, std::move(resources->dispatch_args));
    ReleaseMetalBuffer(adapter, std::move(resources->status));
  }
  delete resources;
}

bool CompileMetalSortPipelines(MetalAdapter &adapter,
                               const rund::kernel::SortKey key,
                               const rund::kernel::u32 block_size,
                               MetalSortPipelines &pipelines) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  if (device == nil || block_size != kMetalSortBlockSize) {
    return false;
  }
  const std::string dispatch_key = SortPipelineKey("dispatch", block_size);
  const std::string histogram_key =
      SortPipelineKey("histogram", key, block_size);
  const std::string prefix_key = SortPipelineKey("prefix", block_size);
  const std::string base_key = SortPipelineKey("base", block_size);
  const std::string scatter_key = SortPipelineKey("scatter", key, block_size);
  LookupMetalSortPipelines(adapter, dispatch_key, histogram_key, prefix_key,
                           base_key, scatter_key, pipelines);
  if (MetalSortPipelinesMatchShape(pipelines)) {
    return true;
  }
  pipelines = {};
  const std::uint64_t create_begin = MonotonicNanoseconds();
  if (!CompileMetalSortPipelineLibrary(adapter, key, block_size, pipelines)) {
    return false;
  }
  StoreMetalSortPipelines(adapter, dispatch_key, histogram_key, prefix_key,
                          base_key, scatter_key, pipelines,
                          MonotonicNanoseconds() - create_begin);
  return true;
#else
  (void)adapter;
  (void)key;
  (void)block_size;
  (void)pipelines;
  return false;
#endif
}

rund::AccelCheck PrepareMetalSort(const rund::AccelDevice &pick,
                                  const rund::kernel::SortDesc &desc,
                                  const rund::kernel::SortPlan &plan,
                                  const rund::kernel::ComputeDomain domain,
                                  const SortBinds &bindings,
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
  rund::kernel::u32 block_size = 0u;
  rund::kernel::u32 block_count = 0u;
  rund::kernel::u64 block_table_bytes = 0u;
  if (!SortShapeOk(desc, plan, bindings) ||
      !SortBlockShapeOk(plan, block_size, block_count, block_table_bytes) ||
      plan.radix_pass_count == 0u || plan.radix_pass_count > 8u) {
    SetMetalLastError(*adapter, "compute_sort_invalid");
    return rund::AccelCheck{false, "compute_sort_invalid"};
  }

  auto *const raw = new MetalSortEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyMetalSortEncodeResources};
  raw->adapter = adapter;
  raw->plan = plan;
  raw->block_size = block_size;
  raw->block_count = block_count;
  raw->signed_order = IsSignedDomain(domain);
  const rund::AccelCheck lookup = LookupMetalSortBuffers(pick, bindings, *raw);
  if (!lookup.ok) {
    SetMetalLastError(*adapter, lookup.reason);
    return lookup;
  }
  const rund::AccelCheck buffers =
      AcquireMetalSortBuffers(*adapter, block_table_bytes, *raw);
  if (!buffers.ok) {
    return buffers;
  }
  const rund::AccelCheck pipelines =
      CompileMetalSortPipelineSet(*adapter, *raw);
  if (!pipelines.ok) {
    return pipelines;
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

rund::AccelCheck EncodeMetalSort(MetalAdapter &adapter,
                                 const std::shared_ptr<void> &resources,
                                 void *const command_encoder) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  MetalSortEncodeState state{};
  const rund::AccelCheck prepared =
      PrepareMetalSortEncodeState(adapter, resources, command_encoder, state);
  if (!prepared.ok) {
    return prepared;
  }
  if (state.bounded) {
    const SortParams params = MetalSortParams(*state.sort, 0u);
    [state.encoder setComputePipelineState:state.dispatch];
    [state.encoder setBuffer:state.dispatch_args offset:0u atIndex:0u];
    [state.encoder setBytes:&params length:sizeof(params) atIndex:1u];
    [state.encoder setBuffer:state.logical_count
                      offset:state.logical_count_offset
                     atIndex:2u];
    [state.encoder setBuffer:state.status offset:0u atIndex:3u];
    [state.encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
    [state.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  }
  for (rund::kernel::u32 pass = 0u; pass < state.sort->plan.radix_pass_count;
       ++pass) {
    EncodeMetalSortPass(state, pass);
  }
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  (void)command_encoder;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

rund::AccelCheck ExecuteMetalSort(const rund::AccelDevice &pick,
                                  const rund::kernel::SortDesc &desc,
                                  const rund::kernel::SortPlan &plan,
                                  const rund::kernel::ComputeDomain domain,
                                  const SortBinds &bindings) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const adapter = static_cast<MetalAdapter *>(pick.backend.context);
  if (!MetalPickOwnsAdapter(pick) || adapter == nullptr ||
      adapter->queue == nullptr) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  std::shared_ptr<void> resources{};
  const rund::AccelCheck prepare =
      PrepareMetalSort(pick, desc, plan, domain, bindings, resources);
  if (!prepare.ok) {
    return prepare;
  }
  CommandRun command{};
  const rund::AccelCheck open = OpenCommand(*adapter, command);
  if (!open.ok) {
    return open;
  }
  const rund::AccelCheck encode =
      EncodeMetalSort(*adapter, resources, (__bridge void *)command.encoder);
  const rund::AccelCheck submit = FinishCommand(*adapter, command, encode);
  if (!submit.ok) {
    return submit;
  }
  return FinishMetalSort(*adapter, resources);
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
