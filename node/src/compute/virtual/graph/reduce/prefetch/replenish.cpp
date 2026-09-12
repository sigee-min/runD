#include "internal.hpp"

namespace rund::compute::detail::graph_reduce {

Status PrefetchController::replenish(const std::uint64_t consumed_batch,
                                     bool &cleanup_failed) noexcept {
  if (state_.pipeline->device->backend == Backend::Cpu) {
    return Status::success();
  }
  // Current-batch dependency-ready inputs share the fixed worker window.
  // Demand-driven Prefix supply resumes after all middle Forecasts retire;
  // speculative future batches cannot occupy either current-demand slot.
  if (parallel_supported()) {
    return Status::success();
  }
  const std::uint64_t distance = run_.active.graph.prefetch_distance();
  if (distance > lanes_.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::uint64_t supply_distance = std::max<std::uint64_t>(1u, distance);
  if (consumed_batch >
          std::numeric_limits<std::uint64_t>::max() - supply_distance ||
      consumed_batch + supply_distance >= batches_) {
    return Status::success();
  }
  return schedule(consumed_batch + supply_distance, distance != 0u,
                  cleanup_failed);
}

} // namespace rund::compute::detail::graph_reduce
