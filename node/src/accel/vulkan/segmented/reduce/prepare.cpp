#include "model.hpp"

#include "../../../segmented/reduce/shape.hpp"
#include "../../../segmented/reduce/vulkan.hpp"

#include "../../buffer/resident/batch.hpp"

#include <array>
#include <limits>
#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

void DestroyVulkanSegmentedReduce(void *const raw) {
  auto *const resources = static_cast<VulkanSegmentedReduceResources *>(raw);
  if (resources == nullptr) {
    return;
  }
  if (resources->adapter != nullptr) {
    ReleaseVulkanBuffer(*resources->adapter, resources->params);
    ReleaseVulkanBuffer(*resources->adapter, resources->block_counts);
    ReleaseVulkanBuffer(*resources->adapter, resources->block_offsets);
    ReleaseVulkanBuffer(*resources->adapter, resources->segment_starts);
    ReleaseVulkanBuffer(*resources->adapter, resources->segment_count);
    ReleaseVulkanBuffer(*resources->adapter, resources->dispatch_args);
    ReleaseVulkanStatus(*resources->adapter, resources->status);
  }
  delete resources;
}

#endif

rund::AccelCheck
PrepareVulkanSegmentedReduce(const rund::AccelDevice &pick,
                             const rund::kernel::SegmentedReduceDesc &desc,
                             const rund::kernel::SegmentedReducePlan &plan,
                             const rund::kernel::ComputeDomain domain,
                             const SegmentedReduceBinds &bindings,
                             std::shared_ptr<void> &resources) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  resources.reset();
  auto *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr || !SegmentedReduceShapeOk(desc, plan, bindings) ||
      plan.element_count > std::numeric_limits<rund::kernel::u32>::max()) {
    return {false, "compute_segmented_reduce_invalid"};
  }
  VulkanResidentBufferResult input{}, heads{}, output{};
  VulkanResidentReq requests[] = {
      {bindings.input, bindings.input_handle, &input},
      {bindings.heads, bindings.heads_handle, &heads},
      {bindings.output, bindings.output_handle, &output},
  };
  LookupVulkanResidentBatch(pick, requests, "compute_resident_id_invalid");
  if (!input.check.ok || !heads.check.ok || !output.check.ok) {
    return {false, !input.check.ok   ? input.check.reason
                   : !heads.check.ok ? heads.check.reason
                                     : output.check.reason};
  }
  auto *const raw = new VulkanSegmentedReduceResources{};
  std::shared_ptr<void> owner{raw, DestroyVulkanSegmentedReduce};
  raw->adapter = adapter;
  raw->plan = plan;
  raw->output = output.device_buffer;
  raw->input_binding = VulkanStorageBindingFor(input.device_buffer, input.ref);
  raw->heads_binding = VulkanStorageBindingFor(heads.device_buffer, heads.ref);
  raw->output_binding =
      VulkanStorageBindingFor(output.device_buffer, output.ref);
  const std::string classify = VulkanSegmentedClassifySource();
  const std::string prefix = VulkanSegmentedPrefixSource();
  const std::string scatter = VulkanSegmentedScatterSource();
  raw->classify =
      AcquireVulkanSegmentedIndex(*adapter, desc, domain, classify.c_str());
  raw->prefix =
      AcquireVulkanSegmentedIndex(*adapter, desc, domain, prefix.c_str());
  raw->scatter =
      AcquireVulkanSegmentedIndex(*adapter, desc, domain, scatter.c_str());
  raw->reduce = AcquireVulkanSegmentedReduce(*adapter, desc, plan, domain);
  const SegmentedReduceLayout layout =
      SegmentedReduceLayoutFor(plan.element_count);
  const VulkanSegmentedReduceParams params{plan.element_count,
                                           layout.block_count};
  if (raw->input_binding.buffer == nullptr ||
      raw->heads_binding.buffer == nullptr ||
      raw->output_binding.buffer == nullptr || raw->classify == nullptr ||
      raw->prefix == nullptr || raw->scatter == nullptr ||
      raw->reduce == nullptr ||
      !CreateVulkanBuffer(*adapter, sizeof(params),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, raw->params) ||
      !UploadVulkanBuffer(raw->params, &params, sizeof(params)) ||
      !CreateVulkanBuffer(*adapter,
                          layout.block_count * sizeof(rund::kernel::u32),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, raw->block_counts,
                          nullptr, VulkanMemoryUse::Scratch) ||
      !CreateVulkanBuffer(
          *adapter, layout.block_count * sizeof(rund::kernel::u32),
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, raw->block_offsets, nullptr,
          VulkanMemoryUse::Scratch) ||
      !CreateVulkanBuffer(
          *adapter, plan.element_count * sizeof(rund::kernel::u32),
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, raw->segment_starts, nullptr,
          VulkanMemoryUse::Scratch) ||
      !CreateVulkanBuffer(*adapter, sizeof(rund::kernel::u32),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          raw->segment_count, nullptr,
                          VulkanMemoryUse::Device) ||
      !CreateVulkanBuffer(*adapter, 3u * sizeof(rund::kernel::u32),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                          raw->dispatch_args, nullptr,
                          VulkanMemoryUse::Device) ||
      !CreateVulkanStatus(*adapter, sizeof(rund::kernel::u32), raw->status) ||
      !AcquireVulkanCollectiveDescriptorSet(*adapter, *raw->classify,
                                            kVulkanSegmentedReduceBindings,
                                            raw->classify_set) ||
      !AcquireVulkanCollectiveDescriptorSet(*adapter, *raw->prefix,
                                            kVulkanSegmentedReduceBindings,
                                            raw->prefix_set) ||
      !AcquireVulkanCollectiveDescriptorSet(*adapter, *raw->scatter,
                                            kVulkanSegmentedReduceBindings,
                                            raw->scatter_set) ||
      !AcquireVulkanCollectiveDescriptorSet(*adapter, *raw->reduce,
                                            kVulkanSegmentedReduceBindings,
                                            raw->reduce_set)) {
    return {false, VulkanLastError(adapter)};
  }
  const auto storage = [](const VulkanBuffer &buffer) {
    return VulkanStorageBindingFor(buffer);
  };
  bool ready = WriteVulkanStorageDescriptorSet(
      *adapter, raw->classify_set,
      std::array<VulkanStorageBinding, kVulkanSegmentedReduceBindings>{
          storage(raw->params), raw->heads_binding, storage(raw->block_counts),
          storage(raw->block_offsets), storage(raw->segment_starts),
          storage(raw->status.device)});
  ready = ready &&
          WriteVulkanStorageDescriptorSet(
              *adapter, raw->prefix_set,
              std::array<VulkanStorageBinding, kVulkanSegmentedReduceBindings>{
                  storage(raw->params), storage(raw->block_counts),
                  storage(raw->block_offsets), storage(raw->segment_count),
                  storage(raw->dispatch_args), storage(raw->status.device)});
  ready = ready &&
          WriteVulkanStorageDescriptorSet(
              *adapter, raw->scatter_set,
              std::array<VulkanStorageBinding, kVulkanSegmentedReduceBindings>{
                  storage(raw->params), raw->heads_binding,
                  storage(raw->block_offsets), storage(raw->segment_starts),
                  storage(raw->segment_count), storage(raw->status.device)});
  ready = ready &&
          WriteVulkanStorageDescriptorSet(
              *adapter, raw->reduce_set,
              std::array<VulkanStorageBinding, kVulkanSegmentedReduceBindings>{
                  storage(raw->params), raw->input_binding,
                  storage(raw->segment_starts), storage(raw->segment_count),
                  raw->output_binding, storage(raw->status.device)});
  if (!ready) {
    return {false, VulkanLastError(adapter)};
  }
  resources = std::move(owner);
  return {true, "ok"};
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)domain;
  (void)bindings;
  (void)resources;
  return {false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
