#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../partition/shape.hpp"
#include "../collective/execute.hpp"
#include "../kernel/pipeline/template.hpp"
#include "encode/scatter.hpp"
#include "local.hpp"
#include "resources/buffers.hpp"
#include "resources/lookup.hpp"

#include <kernel/program/compute/scan/plan.hpp>

#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
void DestroyVulkanPartitionEncodeResources(void *const raw) {
  auto *const resources = static_cast<VulkanPartitionEncodeResources *>(raw);
  if (resources == nullptr) {
    return;
  }
  VulkanAdapter *const adapter = resources->adapter;
  if (adapter != nullptr) {
    resources->false_scan_resources.reset();
    ReleaseVulkanBuffer(*adapter, resources->params);
    ReleaseVulkanBuffer(*adapter, resources->false_bits);
    ReleaseVulkanBuffer(*adapter, resources->false_offsets);
    ReleaseVulkanBuffer(*adapter, resources->false_totals);
    ReleaseVulkanStatus(*adapter, resources->false_status);
  }
  delete resources;
}
#endif

rund::AccelCheck PrepareVulkanPartition(const rund::AccelDevice &pick,
                                        const rund::kernel::PartitionDesc &desc,
                                        const rund::kernel::PartitionPlan &plan,
                                        const PartitionBinds &bindings,
                                        std::shared_ptr<void> &resources,
                                        const VulkanKernelImmutablePipelines
                                            *const pipelines) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  resources.reset();
  auto *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return rund::AccelCheck{false, "accel_vulkan_unavailable"};
  }
  SetVulkanLastError(*adapter, "ok");
  if (!PartitionShapeOk(desc, plan, bindings)) {
    SetVulkanLastError(*adapter, "compute_partition_invalid");
    return rund::AccelCheck{false, "compute_partition_invalid"};
  }
  const PartitionBufferLookup lookup = LookupPartitionBuffers(pick, bindings);
  if (!PartitionLookupOk(lookup)) {
    const char *const reason = PartitionLookupReason(lookup);
    SetVulkanLastError(*adapter, reason);
    return rund::AccelCheck{false, reason};
  }

  auto *const raw = new VulkanPartitionEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyVulkanPartitionEncodeResources};
  raw->adapter = adapter;
  raw->plan = plan;
  raw->scan_desc = rund::kernel::ScanDesc{
      .op = rund::kernel::ScanOp::ExclusiveSum,
      .element = rund::kernel::ScanElement::U32,
      .element_count = plan.element_count,
      .block_size = block::VulkanPartition,
  };
  raw->scan_plan = rund::kernel::PlanScan(raw->scan_desc);
  raw->flags = lookup.flags.device_buffer;
  raw->values = lookup.values.device_buffer;
  raw->output = lookup.output.device_buffer;
  raw->flags_ref = lookup.flags.ref;
  raw->values_ref = lookup.values.ref;
  raw->output_ref = lookup.output.ref;
  const std::uint32_t tuple_count = raw->scan_plan.pass_count == 1u ? 3u : 5u;
  raw->classify_pipeline =
      pipelines == nullptr
          ? AcquirePartitionPipeline(*adapter, desc, PartitionStage::Classify)
          : pipelines->borrow(rund::kernel::NodeKind::Partition, tuple_count,
                              0u, kPartitionClassifyDescriptorCount, 1u);
  raw->scatter_pipeline =
      pipelines == nullptr
          ? AcquirePartitionPipeline(*adapter, desc, PartitionStage::Scatter)
          : pipelines->borrow(rund::kernel::NodeKind::Partition, tuple_count,
                              1u, kPartitionScatterDescriptorCount, 1u);
  const PartitionParams params_value{plan.element_count};
  if (!raw->scan_plan.ok || raw->classify_pipeline == nullptr ||
      raw->scatter_pipeline == nullptr ||
      !CreateVulkanPartitionScratchBuffers(*adapter, *raw) ||
      !UploadVulkanBuffer(raw->params, &params_value, sizeof(params_value)) ||
      !PrepareVulkanScanBuffers(*adapter, raw->scan_desc, raw->scan_plan,
                                rund::kernel::ComputeDomain::U32,
                                raw->false_bits, raw->false_offsets,
                                raw->false_bits, raw->false_totals,
                                raw->false_status, raw->false_scan_resources,
                                0u, 0u, 0u, 0u, 0u, 0u, pipelines, 2u) ||
      !CreateVulkanPartitionDescriptorSets(*adapter, *raw)) {
    return rund::AccelCheck{false, VulkanLastError(adapter)};
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
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck EncodeVulkanPartition(VulkanAdapter &adapter,
                                       const std::shared_ptr<void> &resources,
                                       void *const command_buffer_raw) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanPartitionEncodeState state{};
  const rund::AccelCheck loaded = LoadVulkanPartitionEncodeState(
      adapter, resources, command_buffer_raw, state);
  if (!loaded.ok) {
    return loaded;
  }
  EncodeVulkanPartitionClassify(*state.partition, state.command,
                                state.workgroups);
  EncodeVulkanPartitionClassifyBarrier(*state.partition, state.command);
  const rund::AccelCheck scan =
      EncodeVulkanPartitionScans(adapter, *state.partition, state.command_raw);
  if (!scan.ok) {
    return scan;
  }
  EncodeVulkanPartitionScatter(*state.partition, state.command,
                               state.workgroups);
  EncodeVulkanPartitionOutputBarrier(*state.partition, state.command);
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  (void)command_buffer_raw;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck ExecuteVulkanPartition(const rund::AccelDevice &pick,
                                        const rund::kernel::PartitionDesc &desc,
                                        const rund::kernel::PartitionPlan &plan,
                                        const PartitionBinds &bindings) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  return ExecuteVulkanCollective(pick, desc, plan, bindings,
                                 [](const rund::AccelDevice &device,
                                    const rund::kernel::PartitionDesc &operation,
                                    const rund::kernel::PartitionPlan &prepared,
                                    const PartitionBinds &resident,
                                    std::shared_ptr<void> &resources) {
                                   return PrepareVulkanPartition(
                                       device, operation, prepared, resident,
                                       resources, nullptr);
                                 },
                                 EncodeVulkanPartition,
                                 FinishVulkanPartition);
#else
  return RejectVulkanCollectiveExecute(pick, desc, plan, bindings);
#endif
}

} // namespace rund::node::accel::detail
