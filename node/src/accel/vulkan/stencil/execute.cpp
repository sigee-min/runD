#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../kernel/scratch.hpp"
#include "../../stencil/shape.hpp"
#include "../collective/execute.hpp"
#include "../kernel/pipeline/template.hpp"
#include "../scratch.hpp"
#include "encode/barrier.hpp"
#include "local.hpp"
#include "resources/buffers.hpp"
#include "resources/lookup.hpp"

#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] VulkanBuffer *
FindVulkanStencilTemporary(VulkanStencilEncodeResources &resources,
                           const RangeTemporaryRole role,
                           const std::uint8_t ordinal) noexcept {
  for (std::size_t index = 0u; index < resources.range.temporary_count();
       ++index) {
    const RangeTemporaryRequirement requirement =
        resources.range.temporary(index);
    if (requirement.role == role && requirement.ordinal == ordinal) {
      return &resources.temporaries[index];
    }
  }
  return nullptr;
}

[[nodiscard]] const VulkanBuffer *
FindVulkanStencilTemporary(const VulkanStencilEncodeResources &resources,
                           const RangeTemporaryRole role,
                           const std::uint8_t ordinal) noexcept {
  return FindVulkanStencilTemporary(
      const_cast<VulkanStencilEncodeResources &>(resources), role, ordinal);
}

[[nodiscard]] bool
PrepareVulkanStencilTemporaries(VulkanAdapter &adapter,
                                VulkanStencilEncodeResources &resources) {
  if (!StencilRangeUsesGlobalScratch(resources.range)) {
    return true;
  }
  VulkanScratch *const active = ActiveVulkanScratch();
  if (active != nullptr) {
    if (!active->valid() || active->page_bytes() == 0u) {
      SetVulkanLastError(adapter, "compute_pipeline_capacity");
      return false;
    }
    const KernelScratchBatchPlan batch = PlanRangeAggregateScratch(
        resources.range, adapter.storage_align, active->page_bytes());
    if (!batch.ok()) {
      SetVulkanLastError(adapter, batch.reason());
      return false;
    }
    for (std::size_t index = 0u; index < resources.range.temporary_count();
         ++index) {
      const RangeTemporaryRequirement requirement =
          resources.range.temporary(index);
      VulkanBuffer &buffer = resources.temporaries[index];
      const KernelScratchPlacement *const placement =
          FindRangeAggregateScratchPlacement(batch, requirement.role,
                                             requirement.ordinal);
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
    const RangeTemporaryRequirement requirement =
        resources.range.temporary(index);
    if (!CreateVulkanBuffer(
            adapter, requirement.bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            resources.temporaries[index], nullptr, VulkanMemoryUse::Scratch)) {
      return false;
    }
  }
  return true;
}

} // namespace

bool VulkanStencilStageScratchBindings(
    const VulkanStencilEncodeResources &resources,
    const std::uint32_t stage_index, const VulkanBuffer *&scratch0,
    const VulkanBuffer *&scratch1) noexcept {
  scratch0 = nullptr;
  scratch1 = nullptr;
  if (!resources.range.ok() || stage_index >= resources.range.stage_count()) {
    return false;
  }
  const RangeAggregateStagePlan stage = resources.range.stage(stage_index);
  const auto require =
      [&](const RangeTemporaryRole role,
          const std::uint8_t ordinal) noexcept -> const VulkanBuffer * {
    const VulkanBuffer *const buffer =
        FindVulkanStencilTemporary(resources, role, ordinal);
    return buffer != nullptr && buffer->buffer != VK_NULL_HANDLE ? buffer
                                                                 : nullptr;
  };
  switch (stage.disposition) {
  case RangeAggregateStageDisposition::Direct:
  case RangeAggregateStageDisposition::SharedHalo:
    return !StencilRangeUsesGlobalScratch(resources.range);
  case RangeAggregateStageDisposition::PrefixBlock:
    scratch0 = require(RangeTemporaryRole::PrefixValues, 0u);
    scratch1 = require(RangeTemporaryRole::BlockSummaries, 0u);
    if (scratch1 == nullptr) {
      scratch1 = scratch0;
    }
    return scratch0 != nullptr && scratch1 != nullptr;
  case RangeAggregateStageDisposition::PrefixSummary:
    if (stage.level == 0u) {
      return false;
    }
    scratch0 = require(RangeTemporaryRole::BlockSummaries,
                       static_cast<std::uint8_t>(stage.level - 1u));
    scratch1 = require(RangeTemporaryRole::BlockSummaries, stage.level);
    if (scratch1 == nullptr) {
      scratch1 = scratch0;
    }
    return scratch0 != nullptr && scratch1 != nullptr;
  case RangeAggregateStageDisposition::PrefixFixup:
    scratch0 = stage.level == 0u
                   ? require(RangeTemporaryRole::PrefixValues, 0u)
                   : require(RangeTemporaryRole::BlockSummaries,
                             static_cast<std::uint8_t>(stage.level - 1u));
    scratch1 = require(RangeTemporaryRole::BlockSummaries, stage.level);
    return scratch0 != nullptr && scratch1 != nullptr;
  case RangeAggregateStageDisposition::PrefixWindow:
    scratch0 = require(RangeTemporaryRole::PrefixValues, 0u);
    scratch1 = scratch0;
    return scratch0 != nullptr;
  case RangeAggregateStageDisposition::BlockPrefixSuffix:
  case RangeAggregateStageDisposition::BlockWindow:
    scratch0 = require(RangeTemporaryRole::ForwardValues, 0u);
    scratch1 = require(RangeTemporaryRole::BackwardValues, 0u);
    return scratch0 != nullptr && scratch1 != nullptr;
  }
  return false;
}

void DestroyVulkanStencilEncodeResources(void *const raw) {
  auto *const resources = static_cast<VulkanStencilEncodeResources *>(raw);
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

rund::AccelCheck PrepareVulkanStencil(
    const rund::AccelDevice &pick, const rund::kernel::StencilDesc &desc,
    const rund::kernel::StencilPlan &plan,
    const rund::kernel::ComputeDomain domain, const StencilBinds &bindings,
    const RangeAggregatePlan &range, std::shared_ptr<void> &resources,
    const VulkanKernelImmutablePipelines *const pipelines) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  resources.reset();
  auto *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return rund::AccelCheck{false, "accel_vulkan_unavailable"};
  }
  SetVulkanLastError(*adapter, "ok");
  if (!StencilShapeOk(desc, plan, bindings)) {
    SetVulkanLastError(*adapter, "compute_stencil_invalid");
    return rund::AccelCheck{false, "compute_stencil_invalid"};
  }
  const StencilGpuShape shape = StencilGpuShapeFromRangeAggregatePlan(range);
  if (!StencilRangeAggregatePlanMatches(plan, domain, range) ||
      !shape.valid() || range.stage_count() == 0u ||
      range.stage_count() > kRangeAggregateStageCapacity) {
    SetVulkanLastError(*adapter, "accel_vulkan_pipeline_unavailable");
    return rund::AccelCheck{false, "accel_vulkan_pipeline_unavailable"};
  }
  for (std::size_t index = 0u; index < range.stage_count(); ++index) {
    if (!RangeAggregateStageDispatchFits(range, index,
                                         adapter->max_dispatch_groups)) {
      SetVulkanLastError(*adapter, "compute_dispatch_overflow");
      return rund::AccelCheck{false, "compute_dispatch_overflow"};
    }
  }
  const StencilBufferLookup lookup = LookupStencilBuffers(pick, bindings);
  if (!StencilLookupOk(lookup)) {
    const char *const reason = StencilLookupReason(lookup);
    SetVulkanLastError(*adapter, reason);
    return rund::AccelCheck{false, reason};
  }

  auto *const raw = new VulkanStencilEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyVulkanStencilEncodeResources};
  raw->adapter = adapter;
  raw->plan = plan;
  raw->range = range;
  raw->shape = shape;
  raw->stage_count = static_cast<std::uint32_t>(range.stage_count());
  raw->input = lookup.input.device_buffer;
  raw->output = lookup.output.device_buffer;
  raw->input_binding =
      VulkanStorageBindingFor(lookup.input.device_buffer, lookup.input.ref);
  raw->output_binding =
      VulkanStorageBindingFor(lookup.output.device_buffer, lookup.output.ref);
  const std::uint32_t descriptor_count = StencilRangeDescriptorCount(range);
  if (raw->input_binding.buffer == nullptr ||
      raw->output_binding.buffer == nullptr ||
      !PrepareVulkanStencilTemporaries(*adapter, *raw)) {
    return rund::AccelCheck{false, VulkanLastError(adapter)};
  }
  for (std::size_t index = 0u; index < range.stage_count(); ++index) {
    VulkanCollectivePipeline *const pipeline =
        pipelines == nullptr
            ? AcquireStencilPipeline(*adapter, desc, domain, range)
            : pipelines->borrow(rund::kernel::NodeKind::Stencil,
                                static_cast<std::uint32_t>(range.stage_count()),
                                index, descriptor_count, 1u);
    const StencilParams params_value = StencilRangeStageParams(range, index);
    if (pipeline == nullptr ||
        (pipelines != nullptr &&
         !VulkanStencilPipelineMatches(*adapter, pipeline, desc, domain,
                                       range)) ||
        !CreateVulkanBuffer(*adapter, sizeof(StencilParams),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            raw->params[index]) ||
        !UploadVulkanBuffer(raw->params[index], &params_value,
                            sizeof(StencilParams))) {
      return rund::AccelCheck{false, VulkanLastError(adapter)};
    }
    raw->pipelines[index] = pipeline;
    if (!CreateVulkanStencilDescriptorSet(*adapter, *raw,
                                          static_cast<std::uint32_t>(index))) {
      return rund::AccelCheck{false, VulkanLastError(adapter)};
    }
  }
  if (raw->stage_count == 0u) {
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
  (void)range;
  (void)resources;
  (void)pipelines;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck EncodeVulkanStencil(VulkanAdapter &adapter,
                                     const std::shared_ptr<void> &resources,
                                     void *const command_buffer_raw) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanStencilEncodeState state{};
  const rund::AccelCheck loaded = LoadVulkanStencilEncodeState(
      adapter, resources, command_buffer_raw, state);
  if (!loaded.ok) {
    return loaded;
  }
  for (std::uint32_t index = 0u; index < state.stencil->stage_count; ++index) {
    EncodeVulkanStencilDispatch(*state.stencil, state.command, index);
    if (index + 1u < state.stencil->stage_count) {
      EncodeVulkanStencilStageBarrier(*state.stencil, state.command, index);
    }
  }
  EncodeVulkanStencilFinishBarrier(*state.stencil, state.command);
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  (void)command_buffer_raw;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck ExecuteVulkanStencil(const rund::AccelDevice &pick,
                                      const rund::kernel::StencilDesc &desc,
                                      const rund::kernel::StencilPlan &plan,
                                      const rund::kernel::ComputeDomain domain,
                                      const StencilBinds &bindings,
                                      const RangeAggregatePlan &range) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  return ExecuteVulkanDomainCollective(
      pick, desc, plan, domain, bindings,
      [&range](const rund::AccelDevice &device,
               const rund::kernel::StencilDesc &operation,
               const rund::kernel::StencilPlan &prepared,
               const rund::kernel::ComputeDomain active_domain,
               const StencilBinds &resident, std::shared_ptr<void> &resources) {
        return PrepareVulkanStencil(device, operation, prepared, active_domain,
                                    resident, range, resources, nullptr);
      },
      EncodeVulkanStencil, FinishVulkanStencil);
#else
  (void)domain;
  (void)range;
  return RejectVulkanCollectiveExecute(pick, desc, plan, bindings);
#endif
}

} // namespace rund::node::accel::detail
