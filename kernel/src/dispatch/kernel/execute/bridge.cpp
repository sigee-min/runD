#include "../local.hpp"

namespace rund::kernel::dispatch::detail {

WorkerStats* PrepareWorkerStats(const bool collect_worker_stats,
                                const std::span<u32> worker_stats_sink,
                                const std::span<u64> worker_start_offset_ns_sink,
                                const std::span<u64> worker_elapsed_ns_sink,
                                const std::span<u64> worker_tail_wait_ns_sink,
                                WorkerStats& stats) {
  if (collect_worker_stats && !worker_stats_sink.empty()) {
    stats.partitions_per_worker_sink = worker_stats_sink;
    for (u32& count : stats.partitions_per_worker_sink) {
      count = 0u;
    }
  }
  if (collect_worker_stats && !worker_start_offset_ns_sink.empty()) {
    stats.worker_start_offset_ns_sink = worker_start_offset_ns_sink;
    for (u64& value : stats.worker_start_offset_ns_sink) {
      value = 0u;
    }
  }
  if (collect_worker_stats && !worker_elapsed_ns_sink.empty()) {
    stats.worker_elapsed_ns_sink = worker_elapsed_ns_sink;
    for (u64& value : stats.worker_elapsed_ns_sink) {
      value = 0u;
    }
  }
  if (collect_worker_stats && !worker_tail_wait_ns_sink.empty()) {
    stats.worker_tail_wait_ns_sink = worker_tail_wait_ns_sink;
    for (u64& value : stats.worker_tail_wait_ns_sink) {
      value = 0u;
    }
  }
  return collect_worker_stats ? &stats : nullptr;
}

WorkerTask BuildDispatchTask(DispatchAdapter& adapter,
                             void* const context,
                             const DispatchFn dispatch,
                             const u32* const ordered_packet_indices,
                             const u32 ordered_packet_count) {
  if (ordered_packet_indices == nullptr) {
    return WorkerTask{
        .context = context,
        .invoke = dispatch,
    };
  }

  adapter = DispatchAdapter{
      .context = context,
      .dispatch = dispatch,
      .ordered_packet_indices = ordered_packet_indices,
      .ordered_packet_count = ordered_packet_count,
  };
  return WorkerTask{
      .context = &adapter,
      .invoke = InvokeMappedPartition,
  };
}

void RecordBackendDispatch(Result& result,
                           const u64 dispatch_cost_ns,
                           const WorkerStats* const stats) {
  result.telemetry.dispatch_cost_ns = dispatch_cost_ns;
  result.telemetry.dispatch_cost_measured = true;
  result.telemetry.backend_execution_cost_ns = dispatch_cost_ns;
  result.telemetry.backend_execution_cost_measured = true;
  result.telemetry.backend_dispatch_count = 1u;
  if (stats != nullptr) {
    CopyWorkerStats(*stats, result.telemetry);
  }
}

} // namespace rund::kernel::dispatch::detail
