#include <accel/check.hpp>
#include <accel/device.hpp>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include "../../kernel/scratch.hpp"
#include "../../range_aggregate/plan.hpp"
#include "../../stencil/shape.hpp"
#include "../command/run.hpp"
#include "../pipeline/template.hpp"
#include "../scratch.hpp"
#include "encode/dispatch.hpp"
#include "local.hpp"
#include "pipeline/store.hpp"
#include "resources/lookup.hpp"

#include <algorithm>
#include <limits>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] rund::kernel::u32
MetalMaximumRangeWidth(id<MTLDevice> device) noexcept {
  return device == nil ? 0u
                       : static_cast<rund::kernel::u32>(std::min<std::uint64_t>(
                             device.maxThreadsPerThreadgroup.width,
                             std::numeric_limits<rund::kernel::u32>::max()));
}

[[nodiscard]] std::uint8_t
MetalRangeWidthMask(const rund::kernel::u32 maximum_width) noexcept {
  std::uint8_t widths = 0u;
  for (const rund::kernel::u32 width : kRangeAggregateWorkgroupWidths) {
    if (width <= maximum_width) {
      widths |= width == 64u    ? kRangeAggregateWidth64Bit
                : width == 128u ? kRangeAggregateWidth128Bit
                                : kRangeAggregateWidth256Bit;
    }
  }
  return widths;
}

} // namespace
#endif

RangeAggregateCapabilities
MetalRangeAggregateCapabilities(const rund::AccelDevice &pick) noexcept {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  if (!MetalPickOwnsAdapter(pick)) {
    return RangeAggregateCapabilities::unavailable();
  }
  auto *const adapter = static_cast<MetalAdapter *>(pick.backend.context);
  if (adapter == nullptr || adapter->device == nullptr) {
    return RangeAggregateCapabilities::unavailable();
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter->device.get();
  const rund::kernel::u32 maximum_width = MetalMaximumRangeWidth(device);
  const std::optional<RangeAggregateCapabilities> capabilities =
      RangeAggregateCapabilities::gpu(
          RangeAggregateSourceVariant::Metal,
          MetalRangeWidthMask(maximum_width), maximum_width,
          kRangeAggregateSharedMemoryReserve, device.maxThreadgroupMemoryLength,
          std::numeric_limits<std::uint32_t>::max(),
          RangeAggregateSupportBit(RangeAggregateSupport::Direct) |
              RangeAggregateSupportBit(RangeAggregateSupport::SharedHalo) |
              RangeAggregateSupportBit(
                  RangeAggregateSupport::PrefixDifference) |
              RangeAggregateSupportBit(
                  RangeAggregateSupport::BlockPrefixSuffix));
  return capabilities.value_or(RangeAggregateCapabilities::unavailable());
#else
  (void)pick;
  return RangeAggregateCapabilities::unavailable();
#endif
}

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] MetalRuntimeBuffer *
FindMetalStencilTemporary(MetalStencilEncodeResources &resources,
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

[[nodiscard]] bool
PrepareMetalStencilTemporaries(MetalAdapter &adapter,
                               MetalStencilEncodeResources &resources) {
  if (!StencilRangeUsesGlobalScratch(resources.range)) {
    return true;
  }
  MetalScratch *const active = ActiveMetalScratch();
  if (active != nullptr) {
    if (!active->valid() || active->page_bytes() == 0u) {
      SetMetalLastError(adapter, "compute_pipeline_capacity");
      return false;
    }
    const KernelScratchBatchPlan batch = PlanRangeAggregateScratch(
        resources.range, adapter.caps.storage_alignment, active->page_bytes());
    if (!batch.ok()) {
      SetMetalLastError(adapter, batch.reason());
      return false;
    }
    for (std::size_t index = 0u; index < resources.range.temporary_count();
         ++index) {
      const RangeTemporaryRequirement requirement =
          resources.range.temporary(index);
      const KernelScratchPlacement *const placement =
          FindRangeAggregateScratchPlacement(batch, requirement.role,
                                             requirement.ordinal);
      if (placement == nullptr) {
        SetMetalLastError(adapter, "compute_pipeline_capacity");
        return false;
      }
      resources.temporaries[index] = active->acquire(*placement);
      if (resources.temporaries[index].buffer == nullptr) {
        SetMetalLastError(adapter, "compute_pipeline_capacity");
        return false;
      }
    }
    return true;
  }
  for (std::size_t index = 0u; index < resources.range.temporary_count();
       ++index) {
    const RangeTemporaryRequirement requirement =
        resources.range.temporary(index);
    resources.temporaries[index] = AcquireMetalBuffer(
        adapter, requirement.bytes, MetalBufferUsage::Scratch);
    if (resources.temporaries[index].buffer == nullptr) {
      SetMetalLastError(adapter, "compute_pipeline_capacity");
      return false;
    }
  }
  return true;
}

} // namespace

bool MetalStencilStageScratchBindings(
    const MetalStencilEncodeResources &resources,
    const std::uint32_t stage_index, const MetalRuntimeBuffer *&scratch0,
    const MetalRuntimeBuffer *&scratch1) noexcept {
  scratch0 = nullptr;
  scratch1 = nullptr;
  if (!resources.range.ok() || stage_index >= resources.range.stage_count()) {
    return false;
  }
  const auto require =
      [&](const RangeTemporaryRole role,
          const std::uint8_t ordinal) noexcept -> const MetalRuntimeBuffer * {
    const MetalRuntimeBuffer *const buffer = FindMetalStencilTemporary(
        const_cast<MetalStencilEncodeResources &>(resources), role, ordinal);
    return buffer != nullptr && buffer->buffer != nullptr ? buffer : nullptr;
  };
  const RangeAggregateStagePlan stage = resources.range.stage(stage_index);
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
#endif

void DestroyMetalStencilEncodeResources(void *const raw) {
  auto *const resources = static_cast<MetalStencilEncodeResources *>(raw);
  if (resources != nullptr && resources->adapter != nullptr) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
    for (MetalRuntimeBuffer &temporary : resources->temporaries) {
      ReleaseMetalBuffer(*resources->adapter, std::move(temporary));
    }
#endif
  }
  delete resources;
}

MetalStencilPipelineAttempt CompileMetalStencilPipeline(
    MetalAdapter &adapter, const rund::kernel::StencilOp op,
    const rund::kernel::StencilElement element,
    const rund::kernel::ComputeDomain domain, const StencilGpuShape shape,
    const RangeAggregatePlan &range, std::shared_ptr<void> &out) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  out.reset();
  const std::string key = StencilPipelineKey(op, element, domain, shape, range);
  if (LookupMetalStencilPipeline(adapter, key, out)) {
    const MetalStencilPipelineAssessment assessment =
        AssessMetalStencilPipeline(adapter, op, element, domain, shape, range,
                                   out);
    if (assessment == MetalStencilPipelineAssessment::Ready) {
      return {MetalStencilPipelineAttemptStatus::Ready, "ok"};
    }
    out.reset();
    return {assessment == MetalStencilPipelineAssessment::Unsupported
                ? MetalStencilPipelineAttemptStatus::Unsupported
                : MetalStencilPipelineAttemptStatus::Failed,
            "accel_metal_pipeline_unavailable"};
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  if (device == nil) {
    return {MetalStencilPipelineAttemptStatus::Failed,
            "accel_metal_pipeline_unavailable"};
  }
  const MetalStencilPipelineAttempt compiled =
      CompileMetalStencilPipelineLibrary(adapter, op, element, domain, shape,
                                         range, out);
  if (compiled.status != MetalStencilPipelineAttemptStatus::Ready) {
    return compiled;
  }
  const MetalNamedPipelinePublishResult published =
      StoreMetalStencilPipeline(adapter, key, out, compiled.create_ns);
  if (published.status != MetalNamedPipelinePublishStatus::Inserted) {
    // This invocation constructed a PSO, so account it even when a concurrent
    // owner won publication or the cache could not publish it. Inserted PSOs
    // are accounted by PublishMetalNamedPipeline itself.
    RecordMetalUncachedPipelineCompile(adapter, compiled.create_ns);
  }
  if (published.status == MetalNamedPipelinePublishStatus::Failed) {
    out.reset();
    return {MetalStencilPipelineAttemptStatus::Failed,
            "compute_pipeline_capacity"};
  }
  out = published.pipeline;
  const MetalStencilPipelineAssessment assessment = AssessMetalStencilPipeline(
      adapter, op, element, domain, shape, range, out);
  if (assessment == MetalStencilPipelineAssessment::Ready) {
    return compiled;
  }
  out.reset();
  return {assessment == MetalStencilPipelineAssessment::Unsupported
              ? MetalStencilPipelineAttemptStatus::Unsupported
              : MetalStencilPipelineAttemptStatus::Failed,
          "accel_metal_pipeline_unavailable"};
#else
  (void)adapter;
  (void)op;
  (void)element;
  (void)domain;
  (void)shape;
  (void)out;
  return {MetalStencilPipelineAttemptStatus::Failed,
          "accel_metal_pipeline_unavailable"};
#endif
}

rund::AccelCheck PrepareMetalStencil(
    const rund::AccelDevice &pick, const rund::kernel::StencilDesc &desc,
    const rund::kernel::StencilPlan &plan,
    const rund::kernel::ComputeDomain domain, const StencilBinds &bindings,
    const RangeAggregatePlan &range, std::shared_ptr<void> &resources,
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
  if (!StencilShapeOk(desc, plan, bindings)) {
    SetMetalLastError(*adapter, "compute_stencil_invalid");
    return rund::AccelCheck{false, "compute_stencil_invalid"};
  }
  const StencilGpuShape shape = StencilGpuShapeFromRangeAggregatePlan(range);
  if (!StencilRangeAggregatePlanMatches(plan, domain, range) ||
      !shape.valid() || range.stage_count() == 0u ||
      range.stage_count() > kRangeAggregateStageCapacity) {
    SetMetalLastError(*adapter, "accel_metal_pipeline_unavailable");
    return rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
  }
  for (std::size_t index = 0u; index < range.stage_count(); ++index) {
    if (!RangeAggregateStageDispatchFits(
            range, index, std::numeric_limits<std::uint32_t>::max())) {
      SetMetalLastError(*adapter, "compute_dispatch_overflow");
      return rund::AccelCheck{false, "compute_dispatch_overflow"};
    }
  }

  auto *const raw = new MetalStencilEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyMetalStencilEncodeResources};
  raw->adapter = adapter;
  raw->plan = plan;
  raw->range = range;
  raw->shape = shape;
  raw->stage_count = static_cast<std::uint32_t>(range.stage_count());
  rund::AccelCheck check =
      LookupMetalStencilResidentBuffers(pick, bindings, *raw);
  if (!check.ok) {
    SetMetalLastError(*adapter, check.reason);
    return check;
  }
  if (!PrepareMetalStencilTemporaries(*adapter, *raw)) {
    return rund::AccelCheck{false, MetalLastError(adapter)};
  }
  if (pipelines != nullptr && !pipelines->ready(raw->stage_count)) {
    check = {false, "accel_metal_pipeline_unavailable"};
  } else {
    for (std::size_t index = 0u; check.ok && index < range.stage_count();
         ++index) {
      std::shared_ptr<void> pipeline;
      if (pipelines != nullptr) {
        pipeline = pipelines->stages[index];
        check =
            AssessMetalStencilPipeline(*adapter, plan.op, plan.element, domain,
                                       shape, range, pipeline) ==
                    MetalStencilPipelineAssessment::Ready
                ? rund::AccelCheck{true, "ok"}
                : rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
      } else {
        const MetalStencilPipelineAttempt attempt = CompileMetalStencilPipeline(
            *adapter, plan.op, plan.element, domain, shape, range, pipeline);
        check = attempt.status == MetalStencilPipelineAttemptStatus::Ready
                    ? rund::AccelCheck{true, "ok"}
                    : rund::AccelCheck{false, attempt.reason};
      }
      if (check.ok) {
        raw->pipelines[index] = std::move(pipeline);
      }
    }
  }
  if (!check.ok) {
    SetMetalLastError(*adapter, check.reason);
    return check;
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
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

rund::AccelCheck EncodeMetalStencil(MetalAdapter &adapter,
                                    const std::shared_ptr<void> &resources,
                                    void *command_encoder) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  MetalStencilCommandState state{};
  const rund::AccelCheck prepared = PrepareMetalStencilCommandState(
      adapter, resources, command_encoder, state);
  if (!prepared.ok) {
    return prepared;
  }
  for (std::uint32_t index = 0u; index < state.stencil->stage_count; ++index) {
    const StencilParams params =
        StencilRangeStageParams(state.stencil->range, index);
    EncodeMetalStencilDispatch(state, index, params);
    if (index + 1u < state.stencil->stage_count &&
        StencilRangeUsesGlobalScratch(state.stencil->range)) {
      [state.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
  }
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  (void)command_encoder;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

rund::AccelCheck ExecuteMetalStencil(const rund::AccelDevice &pick,
                                     const rund::kernel::StencilDesc &desc,
                                     const rund::kernel::StencilPlan &plan,
                                     const rund::kernel::ComputeDomain domain,
                                     const StencilBinds &bindings,
                                     const RangeAggregatePlan &range) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const adapter = static_cast<MetalAdapter *>(pick.backend.context);
  if (!MetalPickOwnsAdapter(pick) || adapter == nullptr ||
      adapter->queue == nullptr) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  std::shared_ptr<void> resources{};
  const rund::AccelCheck prepare =
      PrepareMetalStencil(pick, desc, plan, domain, bindings, range, resources);
  if (!prepare.ok) {
    return prepare;
  }

  CommandRun command{};
  const rund::AccelCheck open = OpenCommand(*adapter, command);
  if (!open.ok) {
    return open;
  }
  const rund::AccelCheck encode =
      EncodeMetalStencil(*adapter, resources, (__bridge void *)command.encoder);
  const rund::AccelCheck submit = FinishCommand(*adapter, command, encode);
  if (!submit.ok) {
    return submit;
  }
  return FinishMetalStencil(*adapter, resources);
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)domain;
  (void)bindings;
  (void)range;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
