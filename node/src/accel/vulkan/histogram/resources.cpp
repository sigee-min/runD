#include <accel/check.hpp>
#include <accel/device.hpp>

#include "local.hpp"

#include "../../histogram/shape.hpp"

#include "../buffer/resident/batch.hpp"
#include "../collective/execute.hpp"

#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
void DestroyVulkanHistogramEncodeResources(void *const raw) {
  auto *const resources = static_cast<VulkanHistogramEncodeResources *>(raw);
  if (resources == nullptr) {
    return;
  }
  if (resources->adapter != nullptr) {
    DestroyVulkanBuffer(*resources->adapter, resources->params);
    ReleaseVulkanStatus(*resources->adapter, resources->status);
  }
  delete resources;
}

rund::AccelCheck PrepareVulkanHistogram(const rund::AccelDevice &pick,
                                        const rund::kernel::HistogramDesc &desc,
                                        const rund::kernel::HistogramPlan &plan,
                                        const HistogramBinds &bindings,
                                        std::shared_ptr<void> &resources) {
  resources.reset();
  auto *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return rund::AccelCheck{false, "accel_vulkan_unavailable"};
  }
  SetVulkanLastError(*adapter, "ok");
  if (!HistogramShapeOk(desc, plan, bindings)) {
    SetVulkanLastError(*adapter, "compute_histogram_invalid");
    return rund::AccelCheck{false, "compute_histogram_invalid"};
  }
  VulkanResidentBufferResult bins{};
  VulkanResidentBufferResult counts{};
  VulkanResidentReq reqs[] = {
      {bindings.bins, bindings.bins_handle, &bins},
      {bindings.counts, bindings.counts_handle, &counts}};
  LookupVulkanResidentBatch(pick, reqs, "compute_resident_id_invalid");
  if (!bins.check.ok || !counts.check.ok || bins.device_buffer == nullptr ||
      counts.device_buffer == nullptr) {
    const char *const reason =
        !bins.check.ok ? bins.check.reason : counts.check.reason;
    SetVulkanLastError(*adapter, reason);
    return rund::AccelCheck{false, reason};
  }

  auto *const raw = new VulkanHistogramEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyVulkanHistogramEncodeResources};
  raw->adapter = adapter;
  raw->plan = plan;
  raw->bins = bins.device_buffer;
  raw->counts = counts.device_buffer;
  raw->bins_binding = VulkanStorageBindingFor(bins.device_buffer, bins.ref);
  raw->counts_binding =
      VulkanStorageBindingFor(counts.device_buffer, counts.ref);
  raw->clear_pipeline = AcquireHistogramPipeline(*adapter, desc, true);
  raw->count_pipeline = AcquireHistogramPipeline(*adapter, desc, false);
  const HistogramParams params_value{plan.element_count, plan.bin_count};
  if (raw->bins_binding.buffer == nullptr ||
      raw->counts_binding.buffer == nullptr || raw->clear_pipeline == nullptr ||
      raw->count_pipeline == nullptr ||
      !CreateVulkanBuffer(*adapter, sizeof(params_value),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, raw->params) ||
      !UploadVulkanBuffer(raw->params, &params_value, sizeof(params_value)) ||
      !CreateVulkanStatus(*adapter, plan.status_bytes, raw->status) ||
      !CreateVulkanHistogramDescriptorSets(*adapter, *raw)) {
    return rund::AccelCheck{false, VulkanLastError(adapter)};
  }
  resources = std::move(owned);
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
