#include <accel/check.hpp>
#include <accel/device.hpp>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include "../../../kernel/backend/run.hpp"
#include "../../../kernel/preparation.hpp"
#include "../../../kernel/scratch.hpp"
#include "../../../range_aggregate/execution/control.hpp"
#include "../../../range_aggregate/plan.hpp"
#include "../../pipeline/template.hpp"
#include "../../scratch.hpp"
#include "../encode/dispatch.hpp"
#include "../local.hpp"
#include "../pipeline/store.hpp"
#include "prepare.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace rund::node::accel::detail {

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

} // namespace

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

} // namespace rund::node::accel::detail
