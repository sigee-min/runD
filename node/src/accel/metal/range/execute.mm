#include <accel/check.hpp>
#include <accel/device.hpp>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include "../../kernel/backend/run.hpp"
#include "../../kernel/preparation.hpp"
#include "../../kernel/scratch.hpp"
#include "../../range_aggregate/plan.hpp"
#include "../pipeline/template.hpp"
#include "../scratch.hpp"
#include "encode/dispatch.hpp"
#include "local.hpp"
#include "pipeline/store.hpp"

#include <algorithm>
#include <cstring>
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

[[nodiscard]] std::string MetalRangeControlKey(const RangePlan &plan) {
  const RangeShape &shape = plan.shape();
  std::string key = "range.control";
  const auto append = [&key](const std::uint64_t value) {
    key.push_back('.');
    key += std::to_string(value);
  };
  append(static_cast<std::uint8_t>(shape.count()));
  append(shape.input_count());
  append(shape.window_size());
  append(shape.stride());
  append(shape.padding());
  append(plan.stage_count());
  for (std::size_t index = 0u; index < plan.stage_count(); ++index) {
    const RangeStagePlan stage = plan.stage(index);
    append(static_cast<std::uint8_t>(stage.disposition));
    append(stage.level);
    append(stage.width);
  }
  return key;
}

[[nodiscard]] std::shared_ptr<void>
AcquireMetalRangeControlPipeline(MetalAdapter &adapter, const RangePlan &plan) {
  const std::string key = MetalRangeControlKey(plan);
  std::shared_ptr<void> pipeline = LookupMetalNamedPipeline(adapter, key);
  if (pipeline != nullptr) {
    return pipeline;
  }
  std::uint64_t source_upper = 0u;
  if (!MetalRangeControlSourceUpperBytes(plan, source_upper)) {
    return {};
  }
  const std::shared_ptr<void> library =
      AcquireMetalLibrary(adapter, MetalRangeControlSource(plan), source_upper);
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  id<MTLLibrary> native = (__bridge id<MTLLibrary>)library.get();
  const std::uint64_t begin = MonotonicNanoseconds();
  if (device == nil || native == nil) {
    return {};
  }
  if (!MakeNamedMetalPipeline(device, native, "rund_range_control", pipeline)) {
    RecordMetalUncachedPipelineCompile(adapter, MonotonicNanoseconds() - begin);
    return {};
  }
  const std::uint64_t create_ns = MonotonicNanoseconds() - begin;
  const MetalNamedPipelinePublishResult published =
      PublishMetalNamedPipeline(adapter, key, pipeline, create_ns);
  if (published.status != MetalNamedPipelinePublishStatus::Inserted) {
    RecordMetalUncachedPipelineCompile(adapter, create_ns);
  }
  return published.status == MetalNamedPipelinePublishStatus::Failed
             ? std::shared_ptr<void>{}
             : published.pipeline;
}

[[nodiscard]] bool
PrepareMetalRangeControl(const rund::AccelDevice &pick,
                         const BoundControl *const bound,
                         MetalRangeResources &resources,
                         const MetalKernelImmutablePipelines *const pipelines) {
  if (!resources.range.shape().resident_counted()) {
    return bound == nullptr || !bound->active();
  }
  if (bound == nullptr || !bound->control.has_count() ||
      bound->control.has_predicate() || bound->count == nullptr ||
      bound->count_handle == nullptr ||
      bound->control.capacity != resources.range.shape().input_count() ||
      (resources.range.shape().count() == RangeCount::U64) !=
          (bound->control.count_source ==
           rund::kernel::GraphControlSource::U64)) {
    return false;
  }
  const std::optional<RangeControlLayout> layout =
      RangeControlLayout::from(resources.range);
  if (!layout.has_value()) {
    return false;
  }
  resources.control_count =
      LookupMetalResidentBuffer(pick, *bound->count, *bound->count_handle);
  if (!resources.control_count.check.ok ||
      resources.control_count.device_buffer == nullptr) {
    return false;
  }
  resources.control_count.ref = *bound->count;
  resources.control = bound->control;
  resources.control_params = AcquireMetalBuffer(
      *resources.adapter, layout->params_bytes(), MetalBufferUsage::Output);
  resources.control_indirect = AcquireMetalBuffer(
      *resources.adapter, layout->indirect_bytes(), MetalBufferUsage::Output);
  resources.control_status =
      AcquireMetalBuffer(*resources.adapter, RangeControlLayout::status_bytes(),
                         MetalBufferUsage::Output);
  resources.control_pipeline = pipelines == nullptr
                                   ? AcquireMetalRangeControlPipeline(
                                         *resources.adapter, resources.range)
                                   : pipelines->control;
  resources.controlled = true;
  resources.indirect =
      !IsPipelinePrivatePreparation(CurrentKernelPreparationMode());
  void *const status = MetalBufferContents(resources.control_status);
  if (resources.control_params.buffer == nullptr ||
      resources.control_indirect.buffer == nullptr ||
      resources.control_status.buffer == nullptr ||
      resources.control_pipeline == nullptr || status == nullptr) {
    return false;
  }
  std::memset(status, 0,
              static_cast<std::size_t>(RangeControlLayout::status_bytes()));
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
    ReleaseMetalBuffer(*resources->adapter,
                       std::move(resources->control_params));
    ReleaseMetalBuffer(*resources->adapter,
                       std::move(resources->control_indirect));
    ReleaseMetalBuffer(*resources->adapter,
                       std::move(resources->control_status));
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
                  const BoundControl *const control,
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
  if (!PrepareMetalRangeControl(pick, control, *raw, pipelines)) {
    check = {false, "compute_range_aggregate_invalid"};
  } else if (pipelines != nullptr &&
             !pipelines->ready(raw->stage_count, raw->controlled)) {
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
  (void)control;
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
  if (state.range->controlled) {
    id<MTLComputePipelineState> control =
        (__bridge id<MTLComputePipelineState>)
            state.range->control_pipeline.get();
    id<MTLBuffer> count =
        (__bridge id<MTLBuffer>)state.range->control_count.device_buffer.get();
    id<MTLBuffer> params =
        (__bridge id<MTLBuffer>)state.range->control_params.buffer.get();
    id<MTLBuffer> indirect =
        (__bridge id<MTLBuffer>)state.range->control_indirect.buffer.get();
    id<MTLBuffer> status =
        (__bridge id<MTLBuffer>)state.range->control_status.buffer.get();
    if (control == nil || count == nil || params == nil || indirect == nil ||
        status == nil) {
      return rund::AccelCheck{false, "accel_metal_command_unavailable"};
    }
    [state.encoder setComputePipelineState:control];
    [state.encoder setBuffer:count
                      offset:static_cast<NSUInteger>(
                                 state.range->control_count.ref.offset_bytes +
                                 state.range->control.count_byte_offset)
                     atIndex:0u];
    [state.encoder setBuffer:params offset:0u atIndex:1u];
    [state.encoder setBuffer:indirect offset:0u atIndex:2u];
    [state.encoder setBuffer:status offset:0u atIndex:3u];
    [state.encoder
              dispatchThreads:MTLSizeMake(state.range->stage_count, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(state.range->stage_count, 1u, 1u)];
    [state.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  }
  for (std::uint32_t index = 0u; index < state.range->stage_count; ++index) {
    const std::optional<RangeParams> params =
        state.range->controlled
            ? std::nullopt
            : RangeStageParamsFor(state.range->range, index);
    if (!state.range->controlled && !params.has_value()) {
      SetMetalLastError(adapter, "compute_range_aggregate_invalid");
      return rund::AccelCheck{false, "compute_range_aggregate_invalid"};
    }
    EncodeMetalRangeStage(state, index,
                          params.has_value() ? &*params : nullptr);
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
