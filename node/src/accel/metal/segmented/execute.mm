#include <accel/check.hpp>
#include <accel/device.hpp>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include "../../segmented/shape.hpp"
#include "../command/run.hpp"
#include "local.hpp"
#include "../pipeline/template.hpp"
#include "resources/pipeline.hpp"

#include <utility>

namespace rund::node::accel::detail {

void DestroyMetalSegmentedScanEncodeResources(void *const raw) {
  auto *const resources = static_cast<MetalSegmentedScanEncodeResources *>(raw);
  if (resources == nullptr) {
    return;
  }
  if (resources->adapter != nullptr) {
    ReleaseMetalBuffer(*resources->adapter, std::move(resources->offsets));
    ReleaseMetalBuffer(*resources->adapter, std::move(resources->first_heads));
    ReleaseMetalBuffer(*resources->adapter, std::move(resources->status));
  }
  delete resources;
}

rund::AccelCheck PrepareMetalSegmentedScan(
    const rund::AccelDevice &pick, const rund::kernel::SegmentedScanDesc &desc,
    const rund::kernel::SegmentedScanPlan &plan,
    const rund::kernel::ComputeDomain domain,
    const SegmentedScanBinds &bindings, std::shared_ptr<void> &resources,
    const MetalKernelImmutablePipelines *const pipelines) {
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
  if (!SegmentedScanShapeOk(desc, plan, bindings) ||
      plan.block_count >
          static_cast<rund::kernel::u64>(~rund::kernel::u32{0u})) {
    SetMetalLastError(*adapter, "compute_segmented_scan_invalid");
    return rund::AccelCheck{false, "compute_segmented_scan_invalid"};
  }

  auto *const raw = new MetalSegmentedScanEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyMetalSegmentedScanEncodeResources};
  raw->adapter = adapter;
  raw->plan = plan;
  rund::AccelCheck check =
      LookupMetalSegmentedScanBuffers(pick, bindings, *raw);
  if (check.ok) {
    check = PrepareMetalSegmentedScanStatusBuffer(*adapter, *raw);
  }
  if (check.ok) {
    if (pipelines != nullptr && pipelines->ready(3u)) {
      raw->block = pipelines->stages[0u];
      raw->prefix = pipelines->stages[1u];
      raw->offset = pipelines->stages[2u];
    } else if (pipelines != nullptr) {
      check = rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
    } else {
      check = PrepareMetalSegmentedScanPipeline(*adapter, plan, domain, *raw);
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
  (void)domain;
  (void)bindings;
  (void)resources;
  (void)pipelines;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

rund::AccelCheck
ExecuteMetalSegmentedScan(const rund::AccelDevice &pick,
                          const rund::kernel::SegmentedScanDesc &desc,
                          const rund::kernel::SegmentedScanPlan &plan,
                          const rund::kernel::ComputeDomain domain,
                          const SegmentedScanBinds &bindings) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const adapter = static_cast<MetalAdapter *>(pick.backend.context);
  if (!MetalPickOwnsAdapter(pick) || adapter == nullptr ||
      adapter->queue == nullptr) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  std::shared_ptr<void> resources{};
  const rund::AccelCheck prepare =
      PrepareMetalSegmentedScan(pick, desc, plan, domain, bindings, resources);
  if (!prepare.ok) {
    return prepare;
  }

  CommandRun command{};
  const rund::AccelCheck open = OpenCommand(*adapter, command);
  if (!open.ok) {
    return open;
  }
  const rund::AccelCheck encode = EncodeMetalSegmentedScan(
      *adapter, resources, (__bridge void *)command.encoder);
  const rund::AccelCheck submit = FinishCommand(*adapter, command, encode);
  if (!submit.ok) {
    return submit;
  }
  return FinishMetalSegmentedScan(*adapter, resources);
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
