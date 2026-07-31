#include "model.hpp"

#include "../../../scatter/reduce/status.hpp"

#include "../../kernel/ops/model.hpp"
#include "../../resident.hpp"

#include <rund/compute/reason.hpp>

#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

rund::AccelCheck
FinishMetalScatterReduce(MetalAdapter &adapter,
                         const std::shared_ptr<void> &resources) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  const auto *const state =
      static_cast<const MetalScatterReduceResources *>(resources.get());
  const auto *const status = state == nullptr
                                 ? nullptr
                                 : static_cast<const std::uint32_t *>(
                                       MetalBufferContents(state->status));
  if (state == nullptr || state->adapter != &adapter || status == nullptr) {
    return {false, "compute_scatter_reduce_buffer_invalid"};
  }
  const rund::AccelCheck check = ScatterReduceStatus(status[0]);
  if (!check.ok) {
    return check;
  }
  RecordMetalDispatches(adapter, state->plan.pass_count);
  return {true, "ok"};
#else
  (void)adapter;
  (void)resources;
  return {false, "accel_metal_unavailable"};
#endif
}

PreparedMemory
MetalScatterReduceStepMemory(const std::shared_ptr<void> &resources,
                             const std::uint64_t budget) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  const auto *const state =
      static_cast<const MetalScatterReduceResources *>(resources.get());
  return state == nullptr ? PreparedMemory{}
                          : MetalBuffersMemory(budget, state->status,
                                               state->indirect, state->counts);
#else
  (void)resources;
  (void)budget;
  return {};
#endif
}

bool DescribeMetalScatterReducePipelineStatus(
    const std::shared_ptr<void> &resources,
    MetalPipelineStatusBindings &bindings) noexcept {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  bindings = {};
  const auto *const state =
      static_cast<const MetalScatterReduceResources *>(resources.get());
  if (state == nullptr || state->status.buffer == nullptr) {
    return false;
  }
  bindings.values[0] = MetalPipelineStatusBinding{
      .buffer = state->status.buffer.get(),
      .bytes = state->status.bytes,
      .limit = state->plan.element_count,
      .work_item_count = state->plan.output_count,
      .reasons = {static_cast<std::uint32_t>(
                      rund::compute::Reason::ScatterReduceCountOutOfRange),
                  static_cast<std::uint32_t>(
                      rund::compute::Reason::ScatterReduceIndexOutOfRange),
                  0u, 0u},
      .indirect_dispatch_count = 2u,
      .encoding = MetalPipelineStatusEncoding::Mapping,
      .telemetry = MetalPipelineStatusTelemetry::IndexedControl,
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
