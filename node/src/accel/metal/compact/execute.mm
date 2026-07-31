#include <accel/check.hpp>
#include <accel/device.hpp>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include "../../compact/shape.hpp"
#include "../command/run.hpp"
#include "encode/block.hpp"
#include "encode/element.hpp"
#include "local.hpp"
#include "pipeline/store.hpp"
#include "resources/buffers.hpp"
#include "resources/lookup.hpp"
#include "resources/pipeline.hpp"
#include "resources/scan.hpp"

#include <utility>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
void DestroyMetalCompactEncodeResources(void *const raw) {
  auto *const resources = static_cast<MetalCompactEncodeResources *>(raw);
  if (resources == nullptr) {
    return;
  }
  if (resources->adapter != nullptr) {
    MetalAdapter &adapter = *resources->adapter;
    ReleaseMetalBuffer(adapter, std::move(resources->offsets));
    ReleaseMetalBuffer(adapter, std::move(resources->flag_bits));
    ReleaseMetalBuffer(adapter, std::move(resources->block_counts));
    ReleaseMetalBuffer(adapter, std::move(resources->block_offsets));
    ReleaseMetalBuffer(adapter, std::move(resources->scan_totals));
    ReleaseMetalBuffer(adapter, std::move(resources->scan_status));
    ReleaseMetalBuffer(adapter, std::move(resources->status));
  }
  delete resources;
}

bool CompileMetalCompactPipelines(MetalAdapter &adapter,
                                  MetalCompactPipelines &pipelines,
                                  const bool status_required,
                                  const bool block_offsets) {
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  if (device == nil) {
    return false;
  }
  LookupMetalCompactPipelines(adapter, pipelines, status_required,
                              block_offsets);
  if (MetalCompactPipelinesReady(pipelines, status_required, block_offsets)) {
    return true;
  }
  const std::uint64_t create_begin = MonotonicNanoseconds();
  if (!CompileMetalCompactPipelineLibrary(adapter, pipelines, status_required,
                                          block_offsets)) {
    return false;
  }
  StoreMetalCompactPipelines(adapter, pipelines, status_required, block_offsets,
                             MonotonicNanoseconds() - create_begin);
  return true;
}
#endif

rund::AccelCheck PrepareMetalCompact(const rund::AccelDevice &pick,
                                     const rund::kernel::CompactDesc &desc,
                                     const rund::kernel::CompactPlan &plan,
                                     const CompactBinds &bindings,
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
  if (!CompactShapeOk(desc, plan, bindings)) {
    SetMetalLastError(*adapter, "compute_compact_invalid");
    return rund::AccelCheck{false, "compute_compact_invalid"};
  }
  auto *const raw = new MetalCompactEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyMetalCompactEncodeResources};
  raw->adapter = adapter;
  raw->plan = plan;
  const rund::AccelCheck lookup =
      LookupMetalCompactBuffers(pick, bindings, *raw);
  if (!lookup.ok) {
    SetMetalLastError(*adapter, lookup.reason);
    return lookup;
  }
  const rund::AccelCheck path_buffers =
      AcquireMetalCompactPathBuffers(*adapter, *raw);
  if (!path_buffers.ok) {
    return path_buffers;
  }
  const rund::AccelCheck scan = PrepareMetalCompactScan(*adapter, *raw);
  if (!scan.ok) {
    return scan;
  }
  const rund::AccelCheck pipelines =
      CompileMetalCompactPipelineSet(*adapter, *raw);
  if (!pipelines.ok) {
    return pipelines;
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

rund::AccelCheck EncodeMetalCompact(MetalAdapter &adapter,
                                    const std::shared_ptr<void> &resources,
                                    void *command_encoder) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  MetalCompactEncodeState state{};
  const rund::AccelCheck prepared = PrepareMetalCompactEncodeState(
      adapter, resources, command_encoder, state);
  if (!prepared.ok) {
    return prepared;
  }
  if (state.compact->block_offset_path) {
    return EncodeMetalCompactBlockPath(adapter, command_encoder, state);
  }
  return EncodeMetalCompactElementPath(adapter, command_encoder, state);
#else
  (void)adapter;
  (void)resources;
  (void)command_encoder;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

rund::AccelCheck ExecuteMetalCompact(const rund::AccelDevice &pick,
                                     const rund::kernel::CompactDesc &desc,
                                     const rund::kernel::CompactPlan &plan,
                                     const CompactBinds &bindings) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const adapter = static_cast<MetalAdapter *>(pick.backend.context);
  if (!MetalPickOwnsAdapter(pick) || adapter == nullptr ||
      adapter->queue == nullptr) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  std::shared_ptr<void> resources{};
  const rund::AccelCheck prepare =
      PrepareMetalCompact(pick, desc, plan, bindings, resources);
  if (!prepare.ok) {
    return prepare;
  }
  CommandRun command{};
  const rund::AccelCheck open = OpenCommand(*adapter, command);
  if (!open.ok) {
    return open;
  }
  const rund::AccelCheck encode =
      EncodeMetalCompact(*adapter, resources, (__bridge void *)command.encoder);
  const rund::AccelCheck submit = FinishCommand(*adapter, command, encode);
  if (!submit.ok) {
    return submit;
  }
  return FinishMetalCompact(*adapter, resources);
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)bindings;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
