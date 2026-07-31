#include "local.hpp"

#include <utility>

namespace rund::kernel::dispatch::detail::serial {

void ApplySingleWorkerStats(const std::span<u32> worker_stats_sink,
                            const bool require_no_allocation,
                            const u32 executed_partitions,
                            Telemetry& telemetry) {
  telemetry.worker_count = 1u;
  telemetry.participating_workers = executed_partitions > 0u ? 1u : 0u;
  telemetry.total_partitions_executed = executed_partitions;
  telemetry.worker_idle_slots = executed_partitions > 0u ? 0u : 1u;
  telemetry.min_partitions_per_worker = executed_partitions;
  telemetry.max_partitions_per_worker = executed_partitions;
  telemetry.worker_partition_imbalance_milli = 0u;
  telemetry.tile_imbalance_milli = 0u;
  if (!worker_stats_sink.empty()) {
    worker_stats_sink[0u] = executed_partitions;
    telemetry.partitions_per_worker.clear();
  } else if (require_no_allocation) {
    telemetry.partitions_per_worker.clear();
  } else {
    telemetry.partitions_per_worker.assign(1u, executed_partitions);
  }
}

Result BuildSerialResult(const bool ok, Telemetry&& telemetry) {
  return Result{
      .ok = ok,
      .failure_reason = ok ? "pass" : "dispatch_failed",
      .telemetry = std::move(telemetry),
  };
}

} // namespace rund::kernel::dispatch::detail::serial
