#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../kernel/scratch.hpp"
#include "../kernel/pipeline/template.hpp"
#include "../scratch.hpp"
#include "encode/barrier.hpp"
#include "local.hpp"

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
  }
  delete resources;
}
#endif

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
rund::AccelCheck
PrepareVulkanRange(const rund::AccelDevice &pick, const RangePlan &range,
                   const VulkanRangeBinds &bindings,
                   const rund::kernel::NodeKind owner_kind,
                   std::shared_ptr<void> &resources,
                   const VulkanKernelImmutablePipelines *const pipelines) {
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
        !params_value.has_value() ||
        !CreateVulkanBuffer(*adapter, sizeof(RangeParams),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            raw->params[index]) ||
        !UploadVulkanBuffer(raw->params[index], &*params_value,
                            sizeof(RangeParams))) {
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
  for (std::uint32_t index = 0u; index < state.range->stage_count; ++index) {
    EncodeVulkanRangeStage(*state.range, state.command, index);
    if (index + 1u < state.range->stage_count) {
      EncodeVulkanRangeBarrier(*state.range, state.command, index);
    }
  }
  EncodeVulkanRangeFinishBarrier(*state.range, state.command);
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
