#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../segmented/shape.hpp"
#include "../collective/execute.hpp"
#include "../kernel/pipeline/template.hpp"
#include "local.hpp"
#include "resources/buffers.hpp"

#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
void DestroyVulkanSegmentedScanEncodeResources(void *const raw) {
  auto *const resources =
      static_cast<VulkanSegmentedScanEncodeResources *>(raw);
  if (resources == nullptr) {
    return;
  }
  VulkanAdapter *const adapter = resources->adapter;
  if (adapter != nullptr) {
    ReleaseVulkanBuffer(*adapter, resources->params);
    ReleaseVulkanBuffer(*adapter, resources->offsets);
    ReleaseVulkanBuffer(*adapter, resources->first_heads);
    ReleaseVulkanStatus(*adapter, resources->status);
  }
  delete resources;
}
#endif

rund::AccelCheck PrepareVulkanSegmentedScan(
    const rund::AccelDevice &pick, const rund::kernel::SegmentedScanDesc &desc,
    const rund::kernel::SegmentedScanPlan &plan,
    const rund::kernel::ComputeDomain domain,
    const SegmentedScanBinds &bindings, std::shared_ptr<void> &resources,
    const VulkanKernelImmutablePipelines *const pipelines) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  resources.reset();
  auto *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return rund::AccelCheck{false, "accel_vulkan_unavailable"};
  }
  SetVulkanLastError(*adapter, "ok");
  const rund::kernel::u64 max_u32 =
      static_cast<rund::kernel::u64>(~rund::kernel::u32{0u});
  if (!SegmentedScanShapeOk(desc, plan, bindings) ||
      plan.element_count > max_u32 || plan.block_count > max_u32) {
    SetVulkanLastError(*adapter, "compute_segmented_scan_invalid");
    return rund::AccelCheck{false, "compute_segmented_scan_invalid"};
  }
  const SegmentedScanBufferLookup lookup =
      LookupSegmentedScanBuffers(pick, bindings);
  if (!SegmentedScanLookupOk(lookup)) {
    const char *const reason = SegmentedScanLookupReason(lookup);
    SetVulkanLastError(*adapter, reason);
    return rund::AccelCheck{false, reason};
  }

  auto *const raw = new VulkanSegmentedScanEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyVulkanSegmentedScanEncodeResources};
  raw->adapter = adapter;
  raw->plan = plan;
  raw->input = lookup.input.device_buffer;
  raw->heads = lookup.heads.device_buffer;
  raw->output = lookup.output.device_buffer;
  raw->input_binding =
      VulkanStorageBindingFor(lookup.input.device_buffer, lookup.input.ref);
  raw->heads_binding =
      VulkanStorageBindingFor(lookup.heads.device_buffer, lookup.heads.ref);
  raw->output_binding =
      VulkanStorageBindingFor(lookup.output.device_buffer, lookup.output.ref);
  raw->dispatch_count = ScanDispatches(plan.pass_count, plan.block_count,
                                       adapter->max_dispatch_groups);
  const std::uint32_t stage_count = plan.pass_count == 1u ? 1u : 3u;
  raw->block =
      pipelines == nullptr
          ? AcquireSegmentedScanPipeline(
                *adapter, desc, domain, VulkanSegmentedScanStage::Block)
          : pipelines->borrow(rund::kernel::NodeKind::SegmentedScan,
                              stage_count, 0u,
                              kSegmentedScanDescriptorCount, 1u);
  raw->prefix =
      plan.pass_count == 2u
          ? (pipelines == nullptr
                 ? AcquireSegmentedScanPipeline(
                       *adapter, desc, domain,
                       VulkanSegmentedScanStage::Prefix)
                 : pipelines->borrow(rund::kernel::NodeKind::SegmentedScan,
                                     stage_count, 1u,
                                     kSegmentedScanDescriptorCount, 1u))
          : nullptr;
  raw->offset =
      plan.pass_count == 2u
          ? (pipelines == nullptr
                 ? AcquireSegmentedScanPipeline(
                       *adapter, desc, domain,
                       VulkanSegmentedScanStage::Offset)
                 : pipelines->borrow(rund::kernel::NodeKind::SegmentedScan,
                                     stage_count, 2u,
                                     kSegmentedScanDescriptorCount, 1u))
          : nullptr;
  const SegmentedScanParams params{
      plan.element_count, plan.block_size, plan.block_count,
      plan.op == rund::kernel::SegmentedScanOp::InclusiveSum ? 1u : 0u, 0u};
  if (raw->input_binding.buffer == nullptr ||
      raw->heads_binding.buffer == nullptr ||
      raw->output_binding.buffer == nullptr || raw->dispatch_count == 0u ||
      raw->block == nullptr ||
      (plan.pass_count == 2u &&
       (raw->prefix == nullptr || raw->offset == nullptr)) ||
      !CreateVulkanSegmentedScanBuffers(*adapter, *raw, params) ||
      !CreateVulkanSegmentedScanDescriptorSet(*adapter, *raw)) {
    return rund::AccelCheck{false, VulkanLastError(adapter)};
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
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck
ExecuteVulkanSegmentedScan(const rund::AccelDevice &pick,
                           const rund::kernel::SegmentedScanDesc &desc,
                           const rund::kernel::SegmentedScanPlan &plan,
                           const rund::kernel::ComputeDomain domain,
                           const SegmentedScanBinds &bindings) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  return ExecuteVulkanCollective(
      pick, desc, plan, bindings,
      [domain](const rund::AccelDevice &device,
               const rund::kernel::SegmentedScanDesc &scan,
               const rund::kernel::SegmentedScanPlan &prepared,
               const SegmentedScanBinds &resident,
               std::shared_ptr<void> &resources) {
        return PrepareVulkanSegmentedScan(device, scan, prepared, domain,
                                          resident, resources);
      },
      EncodeVulkanSegmentedScan, FinishVulkanSegmentedScan);
#else
  (void)domain;
  return RejectVulkanCollectiveExecute(pick, desc, plan, bindings);
#endif
}

} // namespace rund::node::accel::detail
