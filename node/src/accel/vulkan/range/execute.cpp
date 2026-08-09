#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../kernel/backend/run.hpp"
#include "../../kernel/scratch.hpp"
#include "../buffer/resident/batch.hpp"
#include "../kernel/pipeline/template.hpp"
#include "../scratch.hpp"
#include "encode/barrier.hpp"
#include "local.hpp"

#include <limits>
#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] VulkanBuffer *
FindVulkanRangeTemp(VulkanRangeResources &resources, const RangeTempRole role,
                    const std::uint8_t ordinal) noexcept {
  for (std::size_t index = 0u; index < resources.range.temporary_count();
       ++index) {
    const RangeTempReq requirement = resources.range.temporary(index);
    if (requirement.role == role && requirement.ordinal == ordinal) {
      return &resources.temporaries[index];
    }
  }
  return nullptr;
}

[[nodiscard]] const VulkanBuffer *
FindVulkanRangeTemp(const VulkanRangeResources &resources,
                    const RangeTempRole role,
                    const std::uint8_t ordinal) noexcept {
  return FindVulkanRangeTemp(const_cast<VulkanRangeResources &>(resources),
                             role, ordinal);
}

[[nodiscard]] bool PrepareVulkanRangeTemps(VulkanAdapter &adapter,
                                           VulkanRangeResources &resources) {
  if (!RangeUsesScratch(resources.range)) {
    return true;
  }
  VulkanScratch *const active = ActiveVulkanScratch();
  if (active != nullptr) {
    if (!active->valid() || active->page_bytes() == 0u) {
      SetVulkanLastError(adapter, "compute_pipeline_capacity");
      return false;
    }
    const KernelScratchBatchPlan batch = PlanRangeScratch(
        resources.range, adapter.storage_align, active->page_bytes());
    if (!batch.ok()) {
      SetVulkanLastError(adapter, batch.reason());
      return false;
    }
    for (std::size_t index = 0u; index < resources.range.temporary_count();
         ++index) {
      const RangeTempReq requirement = resources.range.temporary(index);
      VulkanBuffer &buffer = resources.temporaries[index];
      const KernelScratchPlacement *const placement =
          FindRangeScratch(batch, requirement.role, requirement.ordinal);
      if (placement == nullptr ||
          !active->acquire(*placement, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           buffer)) {
        return false;
      }
    }
    return true;
  }
  for (std::size_t index = 0u; index < resources.range.temporary_count();
       ++index) {
    const RangeTempReq requirement = resources.range.temporary(index);
    if (!CreateVulkanBuffer(
            adapter, requirement.bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            resources.temporaries[index], nullptr, VulkanMemoryUse::Scratch)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool PrepareVulkanRangeControl(
    const rund::AccelDevice &pick, const BoundControl &bound,
    const KernelPreparationMode mode, VulkanRangeResources &resources,
    const VulkanKernelImmutablePipelines *const pipelines) {
  const RangePlan &range = resources.range;
  const RangeCount expected = range.shape().count();
  const rund::kernel::GraphControlSource count_source =
      bound.control.count_source;
  const bool count_type_matches =
      (expected == RangeCount::U32 &&
       count_source == rund::kernel::GraphControlSource::U32) ||
      (expected == RangeCount::U64 &&
       count_source == rund::kernel::GraphControlSource::U64);
  if (!range.shape().resident_counted() || !bound.control.has_count() ||
      bound.control.has_predicate() || !count_type_matches ||
      bound.control.capacity != range.shape().input_count() ||
      bound.count == nullptr || bound.count_handle == nullptr ||
      *bound.count_handle == nullptr) {
    SetVulkanLastError(*resources.adapter, "compute_window_invalid");
    return false;
  }

  VulkanResidentReq count_req[]{
      {bound.count, bound.count_handle, &resources.control_count}};
  LookupVulkanResidentBatch(pick, count_req, "compute_resident_id_invalid");
  const std::uint64_t count_bytes = expected == RangeCount::U64 ? 8u : 4u;
  const rund::kernel::ResidentBufferRef &ref = *bound.count;
  const std::uint64_t alignment = resources.adapter->storage_align;
  std::uint64_t absolute = 0u;
  if (!resources.control_count.check.ok ||
      resources.control_count.device_buffer == nullptr || alignment == 0u ||
      !rund::kernel::checked::add(ref.offset_bytes,
                                  bound.control.count_byte_offset, absolute) ||
      absolute > ref.bytes || count_bytes > ref.bytes - absolute ||
      absolute > resources.control_count.device_buffer->bytes ||
      count_bytes > resources.control_count.device_buffer->bytes - absolute) {
    SetVulkanLastError(*resources.adapter, "compute_window_invalid");
    return false;
  }
  const std::uint64_t count_base = absolute - absolute % alignment;
  const std::uint64_t count_range = absolute + count_bytes - count_base;
  const std::uint64_t count_word = (absolute - count_base) / 4u;
  if (count_range == 0u || count_range > resources.adapter->storage_limit ||
      count_word > std::numeric_limits<std::uint32_t>::max()) {
    SetVulkanLastError(*resources.adapter, "compute_window_invalid");
    return false;
  }

  std::uint64_t stride = sizeof(RangeParams);
  const std::uint64_t remainder = stride % alignment;
  if (remainder != 0u &&
      !rund::kernel::checked::add(stride, alignment - remainder, stride)) {
    SetVulkanLastError(*resources.adapter, "compute_pipeline_capacity");
    return false;
  }
  std::uint64_t params_bytes = 0u;
  std::uint64_t indirect_bytes = 0u;
  if (!rund::kernel::checked::mul(stride, range.stage_count(), params_bytes) ||
      !rund::kernel::checked::mul(sizeof(RangeIndirect), range.stage_count(),
                                  indirect_bytes) ||
      stride / sizeof(std::uint32_t) >
          std::numeric_limits<std::uint32_t>::max() ||
      params_bytes / sizeof(std::uint32_t) >
          std::numeric_limits<std::uint32_t>::max() ||
      params_bytes > std::numeric_limits<VkDeviceSize>::max() ||
      indirect_bytes > std::numeric_limits<VkDeviceSize>::max()) {
    SetVulkanLastError(*resources.adapter, "compute_pipeline_capacity");
    return false;
  }

  resources.control_pipeline =
      pipelines == nullptr
          ? AcquireVulkanRangeControlPipeline(*resources.adapter, range)
          : pipelines->borrow_control(rund::kernel::NodeKind::Window, 4u, 1u);
  if (resources.control_pipeline == nullptr ||
      (pipelines != nullptr &&
       !VulkanRangeControlPipelineMatches(*resources.adapter,
                                          resources.control_pipeline, range)) ||
      !CreateVulkanBuffer(
          *resources.adapter, params_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          resources.control_params, nullptr, VulkanMemoryUse::Device) ||
      !CreateVulkanBuffer(*resources.adapter, indirect_bytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                          resources.control_indirect, nullptr,
                          VulkanMemoryUse::Device) ||
      !CreateVulkanStatus(*resources.adapter,
                          RangeControlLayout::status_bytes(),
                          resources.control_status, mode) ||
      !AcquireVulkanCollectiveDescriptorSet(*resources.adapter,
                                            *resources.control_pipeline, 4u,
                                            resources.control_descriptor)) {
    return false;
  }
  const std::array<VulkanStorageBinding, 4u> control_bindings{
      VulkanStorageBinding{resources.control_count.device_buffer,
                           static_cast<VkDeviceSize>(count_base),
                           static_cast<VkDeviceSize>(count_range)},
      VulkanStorageBindingFor(resources.control_params),
      VulkanStorageBindingFor(resources.control_indirect),
      VulkanStorageBindingFor(resources.control_status.device),
  };
  if (!WriteVulkanStorageDescriptorSet(
          *resources.adapter, resources.control_descriptor, control_bindings)) {
    return false;
  }
  resources.control = bound.control;
  resources.control_param_stride = static_cast<VkDeviceSize>(stride);
  resources.control_push.count_word = static_cast<std::uint32_t>(count_word);
  resources.control_push.param_stride_words =
      static_cast<std::uint32_t>(stride / sizeof(std::uint32_t));
  resources.controlled = true;
  return true;
}

} // namespace

bool VulkanRangeScratch(const VulkanRangeResources &resources,
                        const std::uint32_t stage_index,
                        const VulkanBuffer *&scratch0,
                        const VulkanBuffer *&scratch1) noexcept {
  scratch0 = nullptr;
  scratch1 = nullptr;
  const std::optional<RangeExec> execution = RangeExec::from(resources.range);
  if (!execution.has_value()) {
    return false;
  }
  const std::optional<RangeScratch> bindings =
      execution->stage_scratch(stage_index);
  if (!bindings.has_value()) {
    return false;
  }
  const auto require =
      [&](const RangeTempSlot slot) noexcept -> const VulkanBuffer * {
    const VulkanBuffer *const buffer =
        FindVulkanRangeTemp(resources, slot.role(), slot.ordinal());
    return buffer != nullptr && buffer->buffer != VK_NULL_HANDLE ? buffer
                                                                 : nullptr;
  };
  if (!bindings->uses_global_scratch()) {
    return !execution->uses_global_scratch();
  }
  scratch0 = require(bindings->first());
  scratch1 = require(bindings->second());
  return scratch0 != nullptr && scratch1 != nullptr;
}

void DestroyVulkanRangeResources(void *const raw) {
  auto *const resources = static_cast<VulkanRangeResources *>(raw);
  if (resources == nullptr) {
    return;
  }
  VulkanAdapter *const adapter = resources->adapter;
  if (adapter != nullptr) {
    for (VulkanBuffer &params : resources->params) {
      ReleaseVulkanBuffer(*adapter, params);
    }
    for (VulkanBuffer &temporary : resources->temporaries) {
      ReleaseVulkanBuffer(*adapter, temporary);
    }
    ReleaseVulkanBuffer(*adapter, resources->control_params);
    ReleaseVulkanBuffer(*adapter, resources->control_indirect);
    ReleaseVulkanStatus(*adapter, resources->control_status);
  }
  delete resources;
}
#endif

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
rund::AccelCheck PrepareVulkanRange(
    const rund::AccelDevice &pick, const RangePlan &range,
    const VulkanRangeBinds &bindings, const rund::kernel::NodeKind owner_kind,
    std::shared_ptr<void> &resources,
    const VulkanKernelImmutablePipelines *const pipelines,
    const BoundControl *const control, const KernelPreparationMode mode) {
  resources.reset();
  auto *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return rund::AccelCheck{false, "accel_vulkan_unavailable"};
  }
  SetVulkanLastError(*adapter, "ok");
  const std::optional<RangeExec> execution = RangeExec::from(range);
  if (!execution.has_value()) {
    SetVulkanLastError(*adapter, "compute_range_aggregate_invalid");
    return rund::AccelCheck{false, "compute_range_aggregate_invalid"};
  }
  for (std::size_t index = 0u; index < range.stage_count(); ++index) {
    if (!execution->stage_dispatch_fits(index, adapter->max_dispatch_groups)) {
      SetVulkanLastError(*adapter, "compute_dispatch_overflow");
      return rund::AccelCheck{false, "compute_dispatch_overflow"};
    }
  }
  auto *const raw = new VulkanRangeResources{};
  std::shared_ptr<void> owned{raw, DestroyVulkanRangeResources};
  raw->adapter = adapter;
  raw->range = range;
  raw->stage_count = static_cast<std::uint32_t>(range.stage_count());
  raw->input = bindings.input();
  raw->output = bindings.output();
  raw->input_binding = bindings.input_binding();
  raw->output_binding = bindings.output_binding();
  const std::uint32_t descriptor_count = execution->descriptor_count();
  if (raw->input_binding.buffer == nullptr ||
      raw->output_binding.buffer == nullptr ||
      !PrepareVulkanRangeTemps(*adapter, *raw)) {
    return rund::AccelCheck{false, VulkanLastError(adapter)};
  }
  if (range.shape().resident_counted()) {
    if (owner_kind != rund::kernel::NodeKind::Window || control == nullptr ||
        !PrepareVulkanRangeControl(pick, *control, mode, *raw, pipelines)) {
      return rund::AccelCheck{false, VulkanLastError(adapter)};
    }
  } else if (control != nullptr && control->active()) {
    SetVulkanLastError(*adapter, "compute_window_invalid");
    return rund::AccelCheck{false, "compute_window_invalid"};
  }
  for (std::size_t index = 0u; index < range.stage_count(); ++index) {
    VulkanCollectivePipeline *const pipeline =
        pipelines == nullptr
            ? AcquireVulkanRangePipeline(*adapter, *execution)
            : pipelines->borrow(owner_kind,
                                static_cast<std::uint32_t>(range.stage_count()),
                                index, descriptor_count, 1u);
    const std::optional<RangeParams> params_value =
        execution->stage_params(index);
    if (pipeline == nullptr ||
        (pipelines != nullptr &&
         !VulkanRangePipelineMatches(*adapter, pipeline, *execution)) ||
        (!raw->controlled &&
         (!params_value.has_value() ||
          !CreateVulkanBuffer(*adapter, sizeof(RangeParams),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              raw->params[index]) ||
          !UploadVulkanBuffer(raw->params[index], &*params_value,
                              sizeof(RangeParams))))) {
      return rund::AccelCheck{false, VulkanLastError(adapter)};
    }
    raw->pipelines[index] = pipeline;
    if (!CreateVulkanRangeDescriptors(*adapter, *raw,
                                      static_cast<std::uint32_t>(index))) {
      return rund::AccelCheck{false, VulkanLastError(adapter)};
    }
  }
  if (raw->stage_count == 0u) {
    return rund::AccelCheck{false, VulkanLastError(adapter)};
  }
  resources = std::move(owned);
  return rund::AccelCheck{true, "ok"};
}

rund::AccelCheck EncodeVulkanRange(VulkanAdapter &adapter,
                                   const std::shared_ptr<void> &resources,
                                   void *const command_buffer_raw) {
  VulkanRangeState state{};
  const rund::AccelCheck loaded =
      LoadVulkanRangeState(adapter, resources, command_buffer_raw, state);
  if (!loaded.ok) {
    return loaded;
  }
  if (state.range->controlled) {
    EncodeVulkanRangeControl(*state.range, state.command);
    EncodeVulkanRangeControlBarrier(*state.range, state.command);
  }
  for (std::uint32_t index = 0u; index < state.range->stage_count; ++index) {
    EncodeVulkanRangeStage(*state.range, state.command, index);
    if (index + 1u < state.range->stage_count) {
      EncodeVulkanRangeBarrier(*state.range, state.command, index);
    }
  }
  if (!EncodeVulkanRangeFinishBarrier(*state.range, state.command)) {
    SetVulkanLastError(adapter, "compute_range_aggregate_invalid");
    return rund::AccelCheck{false, "compute_range_aggregate_invalid"};
  }
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
