#include "encode.hpp"

#include "../../../kernel/backend/execute.hpp"
#include "../../number.hpp"
#include "../local.hpp"

#include <algorithm>

namespace rund::node::accel::detail::metalbatch {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] rund::AccelCheck
packed(MetalAdapter &adapter, MetalMapEncodeResources *const map,
       const BatchMapView &view, const BatchMapGroup &group,
       const Workspace &workspace, id<MTLComputeCommandEncoder> encoder) {
  id<MTLComputePipelineState> pipeline =
      map == nullptr
          ? nil
          : map->prepared == nullptr
                ? nil
                : (__bridge id<MTLComputePipelineState>)
                      map->prepared->pipeline.get();
  if (map == nullptr || map->adapter != &adapter || pipeline == nil ||
      encoder == nil || map->param.buffer == nullptr ||
      !rund::kernel::checked::mul(view.tiles, group.count)) {
    return rund::AccelCheck{false, "compute_batch_layout_invalid"};
  }
  const std::uint64_t total = view.tiles * group.count;
  NSUInteger threads = 0u;
  if (!ToNSUInteger(total, threads)) {
    return rund::AccelCheck{false, "compute_dispatch_overflow"};
  }
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:(__bridge id<MTLBuffer>)map->param.buffer.get()
              offset:0u
             atIndex:0u];
  for (std::size_t index = 0u; index < view.input_count; ++index) {
    NSUInteger offset = 0u;
    if (!ToNSUInteger(group.inputs[index], offset)) {
      return rund::AccelCheck{false, "compute_dispatch_overflow"};
    }
    [encoder setBuffer:(__bridge id<MTLBuffer>)workspace.input.buffer.get()
                offset:offset
               atIndex:static_cast<NSUInteger>(index + 1u)];
  }
  for (std::size_t index = 0u; index < view.output_count; ++index) {
    NSUInteger offset = 0u;
    if (!ToNSUInteger(group.outputs[index], offset)) {
      return rund::AccelCheck{false, "compute_dispatch_overflow"};
    }
    [encoder setBuffer:(__bridge id<MTLBuffer>)workspace.output.buffer.get()
                offset:offset
               atIndex:static_cast<NSUInteger>(view.input_count + index + 1u)];
  }
  const NSUInteger width = std::max<NSUInteger>(
      1u,
      std::min<NSUInteger>(threads, [pipeline maxTotalThreadsPerThreadgroup]));
  [encoder dispatchThreads:MTLSizeMake(threads, 1u, 1u)
      threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
  return rund::AccelCheck{true, "ok"};
}

[[nodiscard]] rund::AccelCheck general(MetalAdapter &adapter,
                                       const BackendBatchEntry &entry,
                                       id<MTLComputeCommandEncoder> encoder) {
  auto *const resources =
      entry.prepared == nullptr
          ? nullptr
          : static_cast<MetalKernelResources *>(entry.prepared->get());
  if (entry.run == nullptr || entry.run->pick == nullptr ||
      resources == nullptr || resources->size() == 0u) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  MetalKernelContext context{};
  const rund::AccelCheck valid =
      ValidateMetalKernelContext(*entry.run->pick, context);
  if (!valid.ok || context.adapter != &adapter) {
    return valid.ok ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
                    : valid;
  }
  for (std::size_t index = 0u; index < resources->size(); ++index) {
    MetalKernelEntry *const step = resources->entry(index);
    if (step == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const rund::AccelCheck reset =
        EncodeMetalResets(*resources, index, encoder);
    if (!reset.ok) {
      return reset;
    }
    if (step->barrier_before && step->resets.empty()) {
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    const rund::AccelCheck encoded =
        EncodeMetalStep(adapter, step->ops, step->resource, encoder);
    if (!encoded.ok) {
      return encoded;
    }
  }
  return rund::AccelCheck{true, "ok"};
}

} // namespace

rund::AccelCheck encode(MetalAdapter &adapter,
                        const std::span<const BackendBatchEntry> entries,
                        const std::span<const BatchMapView> views,
                        const BatchMapPlan &plan, const Maps &maps,
                        const Workspace *const workspace, CommandRun &command) {
  rund::AccelCheck result{true, "ok"};
  for (std::size_t group_index = 0u; result.ok && group_index < plan.size;
       ++group_index) {
    const BatchMapGroup &group = plan.groups[group_index];
    if (group.packed) {
      if (workspace == nullptr) {
        result = rund::AccelCheck{false, "compute_batch_layout_invalid"};
        break;
      }
      result = packed(adapter, maps[group.begin], views[group.begin], group,
                      *workspace, command.encoder);
      continue;
    }
    for (std::size_t entry_index = group.begin;
         result.ok && entry_index < group.begin + group.count; ++entry_index) {
      result = general(adapter, entries[entry_index], command.encoder);
    }
  }
  CloseCommand(command);
  return result;
}

#endif

} // namespace rund::node::accel::detail::metalbatch
