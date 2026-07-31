#include "model.hpp"

#include "../../../kernel/memory.hpp"
#include "../../../segmented/reduce/metal.hpp"
#include "../../../segmented/reduce/status.hpp"

#include "../../kernel/ops/model.hpp"

#include <rund/compute/reason.hpp>

namespace rund::node::accel::detail {

rund::AccelCheck
FinishMetalSegmentedReduce(MetalAdapter &adapter,
                           const std::shared_ptr<void> &resources) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const state =
      static_cast<MetalSegmentedReduceResources *>(resources.get());
  const auto *const status = state == nullptr
                                 ? nullptr
                                 : static_cast<const rund::kernel::u32 *>(
                                       MetalBufferContents(state->status));
  if (state == nullptr || state->adapter != &adapter || status == nullptr) {
    return {false, "compute_segmented_reduce_invalid"};
  }
  const rund::AccelCheck check = SegmentedReduceStatus(*status);
  if (!check.ok) {
    SetMetalLastError(adapter, check.reason);
    return check;
  }
  RecordMetalDispatches(adapter, 4u);
  SetMetalLastError(adapter, "ok");
  return {true, "ok"};
#else
  (void)adapter;
  (void)resources;
  return {false, "accel_metal_unavailable"};
#endif
}

PreparedMemory
MetalSegmentedReduceMemory(const std::shared_ptr<void> &resources,
                           const std::uint64_t budget) noexcept {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  const auto *const state =
      static_cast<const MetalSegmentedReduceResources *>(resources.get());
  return state == nullptr
             ? PreparedMemory{}
             : MetalBuffersMemory(budget, state->block_counts,
                                  state->block_offsets, state->segment_starts,
                                  state->segment_count, state->dispatch_args,
                                  state->status);
#else
  (void)resources;
  (void)budget;
  return {};
#endif
}

bool DescribeMetalSegmentedReducePipelineStatus(
    const std::shared_ptr<void> &resources,
    MetalPipelineStatusBindings &bindings) noexcept {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  bindings = {};
  const auto *const state =
      static_cast<const MetalSegmentedReduceResources *>(resources.get());
  if (state == nullptr || state->status.buffer == nullptr ||
      state->status.bytes < sizeof(std::uint32_t)) {
    return false;
  }
  bindings.values[0] = MetalPipelineStatusBinding{
      .buffer = state->status.buffer.get(),
      .bytes = state->status.bytes,
      .reasons = {static_cast<std::uint32_t>(
                      rund::compute::Reason::SegmentedReduceSegmentInvalid),
                  static_cast<std::uint32_t>(
                      rund::compute::Reason::SegmentedReduceSumOverflow),
                  static_cast<std::uint32_t>(
                      rund::compute::Reason::SegmentedReduceCountOverflow)},
      .encoding = MetalPipelineStatusEncoding::SegmentedReduce,
  };
  bindings.size = 1u;
  return true;
#else
  (void)resources;
  (void)bindings;
  return false;
#endif
}

} // namespace rund::node::accel::detail
