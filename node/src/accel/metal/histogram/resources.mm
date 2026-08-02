#include <accel/check.hpp>
#include <accel/device.hpp>

#include "local.hpp"

#include "../../histogram/shape.hpp"

#include "../buffer/resident/batch.hpp"
#include "../pipeline/template.hpp"

#include <cstring>
#include <utility>

namespace rund::node::accel::detail {

void DestroyMetalHistogramEncodeResources(void *const raw) {
  auto *const resources = static_cast<MetalHistogramEncodeResources *>(raw);
  if (resources == nullptr) {
    return;
  }
  if (resources->adapter != nullptr) {
    ReleaseMetalBuffer(*resources->adapter, std::move(resources->status));
  }
  delete resources;
}

rund::AccelCheck
PrepareMetalHistogram(const rund::AccelDevice &pick,
                      const rund::kernel::HistogramDesc &desc,
                      const rund::kernel::HistogramPlan &plan,
                      const HistogramBinds &bindings,
                      std::shared_ptr<void> &resources,
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
  if (!HistogramShapeOk(desc, plan, bindings)) {
    SetMetalLastError(*adapter, "compute_histogram_invalid");
    return rund::AccelCheck{false, "compute_histogram_invalid"};
  }
  auto *const raw = new MetalHistogramEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyMetalHistogramEncodeResources};
  raw->adapter = adapter;
  raw->plan = plan;
  MetalResidentReq reqs[] = {
      {bindings.bins, bindings.bins_handle, &raw->bins},
      {bindings.counts, bindings.counts_handle, &raw->counts}};
  LookupMetalResidentBatch(pick, reqs, "accel_metal_resident_id_unavailable");
  if (!raw->bins.check.ok || !raw->counts.check.ok ||
      raw->bins.device_buffer == nullptr ||
      raw->counts.device_buffer == nullptr) {
    const char *const reason =
        !raw->bins.check.ok ? raw->bins.check.reason : raw->counts.check.reason;
    SetMetalLastError(*adapter, reason);
    return rund::AccelCheck{false, reason};
  }
  raw->status = AcquireMetalBuffer(*adapter, plan.status_bytes,
                                   MetalBufferUsage::Output);
  bool pipeline_ready = false;
  if (pipelines == nullptr) {
    pipeline_ready = CompileMetalHistogramPipelines(*adapter, raw->pipelines);
  } else if (pipelines->ready(2u)) {
    raw->pipelines.clear = pipelines->stages[0u];
    raw->pipelines.count = pipelines->stages[1u];
    pipeline_ready = true;
  }
  if (MetalBufferContents(raw->status) == nullptr || !pipeline_ready) {
    SetMetalLastError(*adapter, "accel_metal_pipeline_unavailable");
    return rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
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

} // namespace rund::node::accel::detail
