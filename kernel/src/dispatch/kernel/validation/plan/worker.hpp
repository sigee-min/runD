#pragma once

#include "../../local.hpp"

#include <cstddef>
#include <span>

namespace rund::kernel::dispatch::detail {

[[nodiscard]] inline bool WorkerStatsSinksTooSmall(
    const bool require_no_allocation,
    const bool collect_worker_stats,
    const std::size_t required_size,
    const std::span<u32> partitions,
    const std::span<u64> start_offsets,
    const std::span<u64> elapsed,
    const std::span<u64> tail_wait) {
  return require_no_allocation && collect_worker_stats &&
         (partitions.size() < required_size ||
          (!start_offsets.empty() && start_offsets.size() < required_size) ||
          (!elapsed.empty() && elapsed.size() < required_size) ||
          (!tail_wait.empty() && tail_wait.size() < required_size));
}

[[nodiscard]] inline bool WorkerStatsSinksTooSmall(const Plan& plan) {
  return WorkerStatsSinksTooSmall(plan.require_no_allocation,
                                  plan.collect_worker_stats,
                                  static_cast<std::size_t>(plan.execution_width),
                                  plan.worker_stats_sink,
                                  plan.worker_start_offset_ns_sink,
                                  plan.worker_elapsed_ns_sink,
                                  plan.worker_tail_wait_ns_sink);
}

[[nodiscard]] inline bool WorkerStatsWouldAllocate(
    const bool require_no_allocation,
    const bool collect_worker_stats,
    const WorkerBackendCapabilities& capabilities) {
  return require_no_allocation && collect_worker_stats &&
         !capabilities.supports_no_alloc_worker_stats;
}

[[nodiscard]] inline bool WorkerStatsWouldAllocate(
    const Plan& plan,
    const WorkerBackendCapabilities& capabilities) {
  return WorkerStatsWouldAllocate(plan.require_no_allocation,
                                  plan.collect_worker_stats,
                                  capabilities);
}

} // namespace rund::kernel::dispatch::detail
