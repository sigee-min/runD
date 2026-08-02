#include "model.hpp"

#include "../../../scatter/reduce/model.hpp"

#include "../../buffer/resident/batch.hpp"
#include "../../kernel/pipeline/template.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

struct ResidentSlice final {
  VulkanStorageBinding binding{};
  std::uint32_t base{};
};

[[nodiscard]] bool ResidentBinding(
    const VulkanAdapter &adapter, const VulkanResidentBufferResult &resident,
    const std::uint64_t element_bytes, ResidentSlice &slice) {
  StorageRange range{};
  if (resident.device_buffer == nullptr || element_bytes == 0u ||
      !PlanStorage(adapter, resident.ref, 0u, resident.ref.count, range) ||
      range.offset % element_bytes != 0u ||
      range.offset / element_bytes >
          std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  slice = ResidentSlice{
      .binding =
          VulkanStorageBinding{resident.device_buffer, range.base, range.bytes},
      .base = static_cast<std::uint32_t>(range.offset / element_bytes),
  };
  return true;
}

[[nodiscard]] bool PrepareDescriptor(
    VulkanAdapter &adapter, VulkanCollectivePipeline &pipeline,
    const std::array<VulkanStorageBinding, kVulkanScatterReduceBindings>
        &bindings,
    VkDescriptorSet &descriptor) {
  if (!AcquireVulkanCollectiveDescriptorSet(
          adapter, pipeline, kVulkanScatterReduceBindings, descriptor)) {
    return false;
  }
  return WriteVulkanStorageDescriptorSet(adapter, descriptor, bindings);
}

} // namespace

void DestroyVulkanScatterReduce(void *const raw) {
  auto *const state = static_cast<VulkanScatterReduceResources *>(raw);
  if (state == nullptr) {
    return;
  }
  if (state->adapter != nullptr) {
    ReleaseVulkanBuffer(*state->adapter, state->params);
    ReleaseVulkanBuffer(*state->adapter, state->indirect);
    ReleaseVulkanBuffer(*state->adapter, state->counts);
    ReleaseVulkanStatus(*state->adapter, state->status);
  }
  delete state;
}

#endif

rund::AccelCheck PrepareVulkanScatterReduce(
    const rund::AccelDevice &pick, const rund::kernel::ScatterReducePlan &plan,
    const ScatterReduceBinds &bindings, const KernelPreparationMode mode,
    std::shared_ptr<void> &resources,
    const VulkanKernelImmutablePipelines *const pipelines) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  resources.reset();
  VulkanAdapter *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr || !plan.ok) {
    return {false, "compute_scatter_reduce_buffer_invalid"};
  }
  auto *const raw = new VulkanScatterReduceResources{};
  std::shared_ptr<void> owned{raw, DestroyVulkanScatterReduce};
  raw->adapter = adapter;
  raw->plan = plan;
  VulkanResidentReq base[] = {
      {bindings.values, bindings.values_handle, &raw->values},
      {bindings.indices, bindings.indices_handle, &raw->indices},
      {bindings.output, bindings.output_handle, &raw->output},
  };
  LookupVulkanResidentBatch(pick, base, "compute_resident_id_invalid");
  if (!raw->values.check.ok || !raw->indices.check.ok ||
      !raw->output.check.ok || raw->values.device_buffer == nullptr ||
      raw->indices.device_buffer == nullptr ||
      raw->output.device_buffer == nullptr) {
    return {false, "compute_scatter_reduce_buffer_invalid"};
  }
  if (plan.count_source != rund::kernel::ComputeCountSource::Descriptor) {
    VulkanResidentReq count[] = {
        {bindings.count, bindings.count_handle, &raw->count}};
    LookupVulkanResidentBatch(pick, count, "compute_resident_id_invalid");
    if (!raw->count.check.ok || raw->count.device_buffer == nullptr) {
      return {false, "compute_scatter_reduce_buffer_invalid"};
    }
  }
  raw->control_pipeline =
      pipelines == nullptr
          ? AcquireVulkanScatterReducePipeline(
                *adapter, plan, VulkanScatterReduceStage::Control)
          : pipelines->borrow(rund::kernel::NodeKind::ScatterReduce, 3u, 0u,
                              kVulkanScatterReduceBindings, 1u);
  raw->init_pipeline =
      pipelines == nullptr
          ? AcquireVulkanScatterReducePipeline(
                *adapter, plan, VulkanScatterReduceStage::Init)
          : pipelines->borrow(rund::kernel::NodeKind::ScatterReduce, 3u, 1u,
                              kVulkanScatterReduceBindings, 1u);
  raw->fold_pipeline =
      pipelines == nullptr
          ? AcquireVulkanScatterReducePipeline(
                *adapter, plan, VulkanScatterReduceStage::Fold)
          : pipelines->borrow(rund::kernel::NodeKind::ScatterReduce, 3u, 2u,
                              kVulkanScatterReduceBindings, 1u);
  ResidentSlice values{};
  ResidentSlice indices{};
  ResidentSlice count{};
  ResidentSlice output{};
  if (!ResidentBinding(*adapter, raw->values, plan.element_bytes, values) ||
      !ResidentBinding(*adapter, raw->indices, sizeof(std::uint32_t), indices) ||
      !ResidentBinding(*adapter, raw->output, plan.element_bytes, output) ||
      (plan.count_source != rund::kernel::ComputeCountSource::Descriptor &&
       !ResidentBinding(*adapter, raw->count, sizeof(std::uint32_t), count))) {
    return {false, "compute_resident_bytes_invalid"};
  }
  const ScatterReduceParams params{
      plan.element_count, plan.output_count,
      static_cast<std::uint32_t>(plan.count_source), 0u, values.base,
      indices.base, count.base, output.base};
  if (raw->control_pipeline == nullptr || raw->init_pipeline == nullptr ||
      raw->fold_pipeline == nullptr ||
      !CreateVulkanBuffer(*adapter, sizeof(params),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, raw->params) ||
      !UploadVulkanBuffer(raw->params, &params, sizeof(params)) ||
      !CreateVulkanBuffer(*adapter, plan.indirect_bytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                          raw->indirect, nullptr, VulkanMemoryUse::Device) ||
      !CreateVulkanBuffer(*adapter, plan.output_count * sizeof(std::uint32_t),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, raw->counts,
                          nullptr, VulkanMemoryUse::Scratch) ||
      !CreateVulkanStatus(*adapter, plan.status_bytes, raw->status, mode)) {
    return {false, VulkanLastError(adapter)};
  }
  const VulkanStorageBinding count_binding =
      plan.count_source == rund::kernel::ComputeCountSource::Descriptor
          ? VulkanStorageBindingFor(raw->params)
          : count.binding;
  const std::array<VulkanStorageBinding, kVulkanScatterReduceBindings>
      descriptor_bindings{VulkanStorageBindingFor(raw->params),
                          values.binding,
                          indices.binding,
                          count_binding,
                          output.binding,
                          VulkanStorageBindingFor(raw->status.device),
                          VulkanStorageBindingFor(raw->indirect),
                          VulkanStorageBindingFor(raw->counts)};
  const bool bindings_ok =
      std::all_of(descriptor_bindings.begin(), descriptor_bindings.end(),
                  [](const VulkanStorageBinding &binding) {
                    return binding.buffer != nullptr && binding.range != 0u;
                  });
  if (!bindings_ok ||
      !PrepareDescriptor(*adapter, *raw->control_pipeline, descriptor_bindings,
                         raw->control_descriptor) ||
      !PrepareDescriptor(*adapter, *raw->init_pipeline, descriptor_bindings,
                         raw->init_descriptor) ||
      !PrepareDescriptor(*adapter, *raw->fold_pipeline, descriptor_bindings,
                         raw->fold_descriptor)) {
    return {false, VulkanLastError(adapter)};
  }
  resources = std::move(owned);
  return {true, "ok"};
#else
  (void)pick;
  (void)plan;
  (void)bindings;
  (void)mode;
  (void)resources;
  (void)pipelines;
  return {false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
