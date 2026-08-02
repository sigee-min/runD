#include "model.hpp"

#include "../../../segmented/reduce/metal.hpp"
#include "../../../segmented/reduce/shape.hpp"
#include "../../pipeline/template.hpp"

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include <utility>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

void DestroyMetalSegmentedReduce(void *const raw) {
  auto *const resources = static_cast<MetalSegmentedReduceResources *>(raw);
  if (resources == nullptr) {
    return;
  }
  if (resources->adapter != nullptr) {
    ReleaseMetalBuffer(*resources->adapter, std::move(resources->block_counts));
    ReleaseMetalBuffer(*resources->adapter,
                       std::move(resources->block_offsets));
    ReleaseMetalBuffer(*resources->adapter,
                       std::move(resources->segment_starts));
    ReleaseMetalBuffer(*resources->adapter,
                       std::move(resources->segment_count));
    ReleaseMetalBuffer(*resources->adapter,
                       std::move(resources->dispatch_args));
    ReleaseMetalBuffer(*resources->adapter, std::move(resources->status));
  }
  delete resources;
}

#endif

rund::AccelCheck
PrepareMetalSegmentedReduce(const rund::AccelDevice &pick,
                            const rund::kernel::SegmentedReduceDesc &desc,
                            const rund::kernel::SegmentedReducePlan &plan,
                            const rund::kernel::ComputeDomain domain,
                            const SegmentedReduceBinds &bindings,
                            std::shared_ptr<void> &resources,
                            const MetalKernelImmutablePipelines *const
                                pipelines) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  resources.reset();
  if (!MetalPickOwnsAdapter(pick) ||
      !SegmentedReduceShapeOk(desc, plan, bindings)) {
    return {false, "compute_segmented_reduce_invalid"};
  }
  auto *const adapter = static_cast<MetalAdapter *>(pick.backend.context);
  if (adapter == nullptr || adapter->device == nullptr) {
    return {false, "accel_metal_unavailable"};
  }
  auto *const raw = new MetalSegmentedReduceResources{};
  std::shared_ptr<void> owner{raw, DestroyMetalSegmentedReduce};
  raw->adapter = adapter;
  raw->plan = plan;
  MetalResidentReq reqs[] = {
      {bindings.input, bindings.input_handle, &raw->input},
      {bindings.heads, bindings.heads_handle, &raw->heads},
      {bindings.output, bindings.output_handle, &raw->output},
  };
  LookupMetalResidentBatch(pick, reqs, "accel_metal_resident_id_unavailable");
  if (!raw->input.check.ok || !raw->heads.check.ok || !raw->output.check.ok) {
    return {false, !raw->input.check.ok   ? raw->input.check.reason
                   : !raw->heads.check.ok ? raw->heads.check.reason
                                          : raw->output.check.reason};
  }
  const SegmentedReduceLayout layout =
      SegmentedReduceLayoutFor(plan.element_count);
  raw->block_counts = AcquireMetalBuffer(
      *adapter, layout.block_count * sizeof(rund::kernel::u64),
      MetalBufferUsage::Scratch);
  raw->block_offsets = AcquireMetalBuffer(
      *adapter, layout.block_count * sizeof(rund::kernel::u64),
      MetalBufferUsage::Scratch);
  raw->segment_starts = AcquireMetalBuffer(
      *adapter, plan.element_count * sizeof(rund::kernel::u64),
      MetalBufferUsage::Scratch);
  raw->segment_count = AcquireMetalBuffer(*adapter, sizeof(rund::kernel::u64),
                                          MetalBufferUsage::Output);
  raw->dispatch_args = AcquireMetalBuffer(
      *adapter, 3u * sizeof(rund::kernel::u32), MetalBufferUsage::Output);
  raw->status = AcquireMetalBuffer(*adapter, sizeof(rund::kernel::u32),
                                   MetalBufferUsage::Output);
  if (MetalBufferContents(raw->block_counts) == nullptr ||
      MetalBufferContents(raw->block_offsets) == nullptr ||
      MetalBufferContents(raw->segment_starts) == nullptr ||
      MetalBufferContents(raw->segment_count) == nullptr ||
      MetalBufferContents(raw->dispatch_args) == nullptr ||
      MetalBufferContents(raw->status) == nullptr) {
    return {false, "accel_metal_buffer_unavailable"};
  }
  rund::AccelCheck pipeline{true, "ok"};
  if (pipelines != nullptr && pipelines->ready(4u)) {
    raw->pipelines.classify = pipelines->stages[0u];
    raw->pipelines.prefix = pipelines->stages[1u];
    raw->pipelines.scatter = pipelines->stages[2u];
    raw->pipelines.reduce = pipelines->stages[3u];
  } else if (pipelines != nullptr) {
    pipeline = {false, "accel_metal_pipeline_unavailable"};
  } else {
    pipeline = CompileMetalSegmentedReduce(*adapter, plan, domain,
                                           raw->pipelines);
  }
  if (!pipeline.ok) {
    return pipeline;
  }
  id<MTLComputePipelineState> reduce =
      (__bridge id<MTLComputePipelineState>)raw->pipelines.reduce.get();
  const NSUInteger width = reduce == nil ? 0u : reduce.threadExecutionWidth;
  if (width == 0u || width > kSegmentedIndexWidth ||
      kSegmentedIndexWidth % width != 0u) {
    return {false, "accel_metal_pipeline_unavailable"};
  }
  raw->segments_per_group = kSegmentedIndexWidth / width;
  resources = std::move(owner);
  return {true, "ok"};
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)domain;
  (void)bindings;
  (void)resources;
  (void)pipelines;
  return {false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
