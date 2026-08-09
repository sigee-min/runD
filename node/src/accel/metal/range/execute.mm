#include <accel/check.hpp>
#include <accel/device.hpp>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include "../../kernel/scratch.hpp"
#include "../../range_aggregate/plan.hpp"
#include "../pipeline/template.hpp"
#include "../scratch.hpp"
#include "encode/dispatch.hpp"
#include "local.hpp"
#include "pipeline/store.hpp"

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
  for (const rund::kernel::u32 width : kRangeWidths) {
    if (width <= maximum_width) {
      widths |= width == 64u    ? kRangeWidth64Bit
                : width == 128u ? kRangeWidth128Bit
                                : kRangeWidth256Bit;
    }
  }
  return widths;
}

} // namespace
#endif

RangeCaps MetalRangeCaps(const rund::AccelDevice &pick) noexcept {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  if (!MetalPickOwnsAdapter(pick)) {
    return RangeCaps::unavailable();
  }
  auto *const adapter = static_cast<MetalAdapter *>(pick.backend.context);
  if (adapter == nullptr || adapter->device == nullptr) {
    return RangeCaps::unavailable();
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter->device.get();
  const rund::kernel::u32 maximum_width = MetalMaximumRangeWidth(device);
  const std::optional<RangeCaps> capabilities = RangeCaps::gpu(
      RangeSource::Metal, MetalRangeWidthMask(maximum_width), maximum_width,
      kRangeSharedReserve, device.maxThreadgroupMemoryLength,
      std::numeric_limits<std::uint32_t>::max(),
      std::numeric_limits<rund::kernel::u64>::max(), device.maxBufferLength,
      RangeSupportBit(RangeSupport::Direct) |
          RangeSupportBit(RangeSupport::SharedHalo) |
          RangeSupportBit(RangeSupport::PrefixDifference) |
          RangeSupportBit(RangeSupport::BlockPrefixSuffix));
  return capabilities.value_or(RangeCaps::unavailable());
#else
  (void)pick;
  return RangeCaps::unavailable();
#endif
}

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] MetalRuntimeBuffer *
FindMetalRangeTemp(MetalRangeResources &resources, const RangeTempRole role,
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

[[nodiscard]] bool PrepareMetalRangeTemps(MetalAdapter &adapter,
                                          MetalRangeResources &resources) {
  if (!RangeUsesScratch(resources.range)) {
    return true;
  }
  MetalScratch *const active = ActiveMetalScratch();
  if (active != nullptr) {
    if (!active->valid() || active->page_bytes() == 0u) {
      SetMetalLastError(adapter, "compute_pipeline_capacity");
      return false;
    }
    const KernelScratchBatchPlan batch = PlanRangeScratch(
        resources.range, adapter.caps.storage_alignment, active->page_bytes());
    if (!batch.ok()) {
      SetMetalLastError(adapter, batch.reason());
      return false;
    }
    for (std::size_t index = 0u; index < resources.range.temporary_count();
         ++index) {
      const RangeTempReq requirement = resources.range.temporary(index);
      const KernelScratchPlacement *const placement =
          FindRangeScratch(batch, requirement.role, requirement.ordinal);
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
    const RangeTempReq requirement = resources.range.temporary(index);
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

bool MetalRangeScratch(const MetalRangeResources &resources,
                       const std::uint32_t stage_index,
                       const MetalRuntimeBuffer *&scratch0,
                       const MetalRuntimeBuffer *&scratch1) noexcept {
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
      [&](const RangeTempSlot slot) noexcept -> const MetalRuntimeBuffer * {
    const MetalRuntimeBuffer *const buffer =
        FindMetalRangeTemp(const_cast<MetalRangeResources &>(resources),
                           slot.role(), slot.ordinal());
    return buffer != nullptr && buffer->buffer != nullptr ? buffer : nullptr;
  };
  if (!bindings->uses_global_scratch()) {
    return !execution->uses_global_scratch();
  }
  scratch0 = require(bindings->first());
  scratch1 = require(bindings->second());
  return scratch0 != nullptr && scratch1 != nullptr;
}
#endif

void DestroyMetalRangeResources(void *const raw) {
  auto *const resources = static_cast<MetalRangeResources *>(raw);
  if (resources != nullptr && resources->adapter != nullptr) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
    for (MetalRuntimeBuffer &temporary : resources->temporaries) {
      ReleaseMetalBuffer(*resources->adapter, std::move(temporary));
    }
#endif
  }
  delete resources;
}

MetalRangeAttempt CompileMetalRange(MetalAdapter &adapter,
                                    const RangeExec &execution,
                                    std::shared_ptr<void> &out) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  out.reset();
  const std::string key = RangePipelineKey(execution);
  if (LookupMetalRangePipeline(adapter, key, out)) {
    const MetalRangeAssessment assessment =
        AssessMetalRange(adapter, execution, out);
    if (assessment == MetalRangeAssessment::Ready) {
      return {MetalRangeAttemptStatus::Ready, "ok"};
    }
    out.reset();
    return {assessment == MetalRangeAssessment::Unsupported
                ? MetalRangeAttemptStatus::Unsupported
                : MetalRangeAttemptStatus::Failed,
            "accel_metal_pipeline_unavailable"};
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  if (device == nil) {
    return {MetalRangeAttemptStatus::Failed,
            "accel_metal_pipeline_unavailable"};
  }
  const MetalRangeAttempt compiled =
      CompileMetalRangeLibrary(adapter, execution, out);
  if (compiled.status != MetalRangeAttemptStatus::Ready) {
    return compiled;
  }
  const MetalNamedPipelinePublishResult published =
      StoreMetalRangePipeline(adapter, key, out, compiled.create_ns);
  if (published.status != MetalNamedPipelinePublishStatus::Inserted) {
    // This invocation constructed a PSO, so account it even when a concurrent
    // owner won publication or the cache could not publish it. Inserted PSOs
    // are accounted by PublishMetalNamedPipeline itself.
    RecordMetalUncachedPipelineCompile(adapter, compiled.create_ns);
  }
  if (published.status == MetalNamedPipelinePublishStatus::Failed) {
    out.reset();
    return {MetalRangeAttemptStatus::Failed, "compute_pipeline_capacity"};
  }
  out = published.pipeline;
  const MetalRangeAssessment assessment =
      AssessMetalRange(adapter, execution, out);
  if (assessment == MetalRangeAssessment::Ready) {
    return compiled;
  }
  out.reset();
  return {assessment == MetalRangeAssessment::Unsupported
              ? MetalRangeAttemptStatus::Unsupported
              : MetalRangeAttemptStatus::Failed,
          "accel_metal_pipeline_unavailable"};
#else
  (void)adapter;
  (void)execution;
  (void)out;
  return {MetalRangeAttemptStatus::Failed, "accel_metal_pipeline_unavailable"};
#endif
}

rund::AccelCheck
PrepareMetalRange(const rund::AccelDevice &pick, const RangePlan &range,
                  const MetalRangeBinds &bindings,
                  std::shared_ptr<void> &resources,
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
  const std::optional<RangeExec> execution = RangeExec::from(range);
  if (!execution.has_value()) {
    SetMetalLastError(*adapter, "compute_range_aggregate_invalid");
    return rund::AccelCheck{false, "compute_range_aggregate_invalid"};
  }
  for (std::size_t index = 0u; index < range.stage_count(); ++index) {
    if (!execution->stage_dispatch_fits(
            index, std::numeric_limits<std::uint32_t>::max())) {
      SetMetalLastError(*adapter, "compute_dispatch_overflow");
      return rund::AccelCheck{false, "compute_dispatch_overflow"};
    }
  }

  auto *const raw = new MetalRangeResources{};
  std::shared_ptr<void> owned{raw, DestroyMetalRangeResources};
  raw->adapter = adapter;
  raw->range = range;
  raw->stage_count = static_cast<std::uint32_t>(range.stage_count());
  raw->input = bindings.input();
  raw->output = bindings.output();
  rund::AccelCheck check{true, "ok"};
  if (!PrepareMetalRangeTemps(*adapter, *raw)) {
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
            AssessMetalRange(*adapter, *execution, pipeline) ==
                    MetalRangeAssessment::Ready
                ? rund::AccelCheck{true, "ok"}
                : rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
      } else {
        const MetalRangeAttempt attempt =
            CompileMetalRange(*adapter, *execution, pipeline);
        check = attempt.status == MetalRangeAttemptStatus::Ready
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
  (void)bindings;
  (void)range;
  (void)resources;
  (void)pipelines;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

rund::AccelCheck EncodeMetalRange(MetalAdapter &adapter,
                                  const std::shared_ptr<void> &resources,
                                  void *command_encoder) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  MetalRangeState state{};
  const rund::AccelCheck prepared =
      LoadMetalRangeState(adapter, resources, command_encoder, state);
  if (!prepared.ok) {
    return prepared;
  }
  for (std::uint32_t index = 0u; index < state.range->stage_count; ++index) {
    const std::optional<RangeParams> params =
        RangeStageParamsFor(state.range->range, index);
    if (!params.has_value()) {
      SetMetalLastError(adapter, "compute_range_aggregate_invalid");
      return rund::AccelCheck{false, "compute_range_aggregate_invalid"};
    }
    EncodeMetalRangeStage(state, index, *params);
    if (index + 1u < state.range->stage_count &&
        RangeUsesScratch(state.range->range)) {
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

} // namespace rund::node::accel::detail
