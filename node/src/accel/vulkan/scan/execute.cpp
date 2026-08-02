#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../buffer/resident/batch.hpp"
#include "encode/pass.hpp"
#include "../kernel/pipeline/template.hpp"
#include "local.hpp"
#include "pipeline.hpp"
#include "resources/validation.hpp"

#include <kernel/program/compute/model.hpp>

#include <mutex>
#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
void DestroyVulkanScanEncodeResources(void *const raw) {
  auto *const resources = static_cast<VulkanScanEncodeResources *>(raw);
  if (resources == nullptr) {
    return;
  }
  VulkanAdapter *const adapter = resources->adapter;
  if (adapter != nullptr && adapter->device != VK_NULL_HANDLE) {
    ReleaseVulkanBuffer(*adapter, resources->params);
  }
  delete resources;
}
#endif

rund::AccelCheck PrepareVulkanScanBuffers(
    VulkanAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan,
    const rund::kernel::ComputeDomain domain, const VulkanBuffer &input,
    const VulkanBuffer &output, const VulkanBuffer &logical_count,
    const VulkanBuffer &totals, VulkanStatus &status,
    std::shared_ptr<void> &resources, const std::uint64_t input_offset,
    const std::uint64_t input_range, const std::uint64_t output_offset,
    const std::uint64_t output_range,
    const std::uint64_t logical_count_offset,
    const std::uint64_t logical_count_range,
    const VulkanKernelImmutablePipelines *const pipelines,
    const std::uint32_t pipeline_offset) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  resources.reset();
  SetVulkanLastError(adapter, "ok");
  const rund::AccelCheck valid = ValidateVulkanScanResources(
      adapter, desc, plan, input, output, totals, status);
  if (!valid.ok) {
    return valid;
  }

  auto *const raw = new VulkanScanEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyVulkanScanEncodeResources};
  raw->adapter = &adapter;
  raw->output = output;
  raw->output_binding = VulkanStorageBindingFor(
      raw->output, static_cast<VkDeviceSize>(output_offset),
      static_cast<VkDeviceSize>(output_range));
  raw->totals = totals;
  raw->status = &status;
  raw->logical_count = VulkanStorageBindingFor(
      logical_count, static_cast<VkDeviceSize>(logical_count_offset),
      static_cast<VkDeviceSize>(logical_count_range));
  const VulkanStorageBinding input_binding =
      VulkanStorageBindingFor(input, static_cast<VkDeviceSize>(input_offset),
                              static_cast<VkDeviceSize>(input_range));
  raw->block_count = plan.block_count;
  raw->pass_count = plan.pass_count;
  raw->dispatch_count = ScanDispatches(plan.pass_count, plan.block_count,
                                       adapter.max_dispatch_groups);

  const ScanParams params_value{
      plan.element_count, plan.block_size, plan.block_count,
      static_cast<rund::kernel::u32>(
          rund::kernel::ComputeCountBytes(plan.count_source) /
          sizeof(rund::kernel::u32)),
      desc.op == rund::kernel::ScanOp::InclusiveSum ? 1u : 0u};
  const std::uint32_t scan_stage_count = plan.pass_count == 1u ? 1u : 3u;
  const std::uint32_t tuple_count = pipeline_offset + scan_stage_count;
  const rund::kernel::NodeKind tuple_kind =
      pipeline_offset == 0u ? rund::kernel::NodeKind::Scan
                            : rund::kernel::NodeKind::Partition;
  raw->block = pipelines == nullptr
                   ? AcquireVulkanScanPipeline(adapter, desc, domain,
                                               VulkanScanStage::Block)
                   : pipelines->borrow(tuple_kind, tuple_count, pipeline_offset,
                                       kScanDescriptorCount, 1u);
  raw->prefix =
      plan.pass_count == 2u
          ? (pipelines == nullptr
                 ? AcquireVulkanScanPipeline(adapter, desc, domain,
                                             VulkanScanStage::Prefix)
                 : pipelines->borrow(tuple_kind, tuple_count,
                                     pipeline_offset + 1u,
                                     kScanDescriptorCount, 1u))
          : nullptr;
  raw->offset =
      plan.pass_count == 2u
          ? (pipelines == nullptr
                 ? AcquireVulkanScanPipeline(adapter, desc, domain,
                                             VulkanScanStage::Offset)
                 : pipelines->borrow(tuple_kind, tuple_count,
                                     pipeline_offset + 2u,
                                     kScanDescriptorCount, 1u))
          : nullptr;
  if (raw->dispatch_count == 0u || raw->block == nullptr ||
      (plan.pass_count == 2u &&
       (raw->prefix == nullptr || raw->offset == nullptr)) ||
      !CreateVulkanBuffer(adapter, sizeof(params_value),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, raw->params) ||
      !UploadVulkanBuffer(raw->params, &params_value, sizeof(params_value)) ||
      !CreateVulkanScanDescriptorSets(adapter, *raw, input_binding,
                                      raw->logical_count)) {
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }
  resources = std::move(owned);
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)desc;
  (void)plan;
  (void)domain;
  (void)input;
  (void)output;
  (void)logical_count;
  (void)totals;
  (void)status;
  (void)resources;
  (void)input_offset;
  (void)input_range;
  (void)output_offset;
  (void)output_range;
  (void)logical_count_offset;
  (void)logical_count_range;
  (void)pipelines;
  (void)pipeline_offset;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck EncodeVulkanScanBuffers(
    VulkanAdapter &adapter, const std::shared_ptr<void> &resources,
    void *const command_buffer_handle) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanScanEncodeState state{};
  const rund::AccelCheck loaded = LoadVulkanScanEncodeState(
      adapter, resources, command_buffer_handle, state);
  if (!loaded.ok) {
    return loaded;
  }
  if (!ResetVulkanStatus(state.command, *state.scan->status,
                         sizeof(rund::kernel::u32))) {
    return rund::AccelCheck{false, "compute_scan_invalid"};
  }
  EncodeVulkanScanBlocks(*state.scan, state.command);
  if (state.scan->pass_count == 2u) {
    EncodeVulkanScanBlockBarrier(*state.scan, state.command);
    EncodeVulkanScanPrefix(*state.scan, state.command);
    EncodeVulkanScanPrefixBarrier(*state.scan, state.command);
    EncodeVulkanScanOffset(*state.scan, state.command);
  }
  const std::array<const VulkanBuffer *, 1u> outputs{&state.scan->output};
  if (!FinishVulkanStatus(state.command, *state.scan->status, outputs)) {
    return rund::AccelCheck{false, "compute_scan_invalid"};
  }
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  (void)command_buffer_handle;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck ExecuteVulkanScan(const rund::AccelDevice &pick,
                                   const rund::kernel::ScanDesc &desc,
                                   const rund::kernel::ScanPlan &plan,
                                   const rund::kernel::ComputeDomain domain,
                                   const ScanBinds &bindings) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return rund::AccelCheck{false, "accel_vulkan_unavailable"};
  }
  std::lock_guard<std::mutex> lock{adapter->mutex};
  SetVulkanLastError(*adapter, "ok");
  if (!ScanShapeOk(desc, plan) || !ScanResidentShapeOk(plan, bindings)) {
    SetVulkanLastError(*adapter, "compute_scan_invalid");
    return rund::AccelCheck{false, "compute_scan_invalid"};
  }
  VulkanResidentBufferResult input{};
  VulkanResidentBufferResult output{};
  VulkanResidentBufferResult logical_count{};
  VulkanResidentReq reqs[] = {
      {bindings.input, bindings.input_handle, &input},
      {bindings.output, bindings.output_handle, &output}};
  LookupVulkanResidentBatch(pick, reqs, "compute_resident_id_invalid");
  if (bindings.logical_count_handle != nullptr) {
    VulkanResidentReq count[] = {{bindings.logical_count,
                                  bindings.logical_count_handle,
                                  &logical_count}};
    LookupVulkanResidentBatch(pick, count, "compute_resident_id_invalid");
  } else {
    logical_count = input;
  }
  if (!input.check.ok || !output.check.ok || !logical_count.check.ok ||
      input.device_buffer == nullptr || output.device_buffer == nullptr ||
      logical_count.device_buffer == nullptr) {
    const char *const reason =
        !input.check.ok ? input.check.reason
                        : (!output.check.ok ? output.check.reason
                                            : logical_count.check.reason);
    SetVulkanLastError(*adapter, reason);
    return rund::AccelCheck{false, reason};
  }
  return ExecuteVulkanScanBuffers(*adapter, desc, plan, domain,
                                  *input.device_buffer, *output.device_buffer,
                                  true, *logical_count.device_buffer);
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)domain;
  (void)bindings;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
