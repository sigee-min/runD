#pragma once

#include <kernel/internal/dispatch/kernel.hpp>

namespace rund::kernel::dispatch::detail {

using internal::Plan;

struct DispatchAdapter {
  void *context = nullptr;
  DispatchFn dispatch = nullptr;
  const u32 *ordered_packet_indices = nullptr;
  u32 ordered_packet_count = 0u;
};

Partition MapPartitionPackets(const DispatchAdapter &adapter,
                              const Partition &partition);
void InvokeMappedPartition(void *raw_context, const Partition &partition);

Telemetry BuildBaseTelemetry(const Plan &plan);
Result FailResult(const Plan &plan, const char *reason);
void CopyWorkerStats(const WorkerStats &stats, Telemetry &telemetry);
WorkerStats *PrepareWorkerStats(bool collect_worker_stats,
                                std::span<u32> worker_stats_sink,
                                std::span<u64> worker_start_offset_ns_sink,
                                std::span<u64> worker_elapsed_ns_sink,
                                std::span<u64> worker_tail_wait_ns_sink,
                                WorkerStats &stats);
WorkerTask BuildDispatchTask(DispatchAdapter &adapter, void *context,
                             DispatchFn dispatch,
                             const u32 *ordered_packet_indices,
                             u32 ordered_packet_count);
void RecordBackendDispatch(Result &result, u64 dispatch_cost_ns,
                           const WorkerStats *stats);

WorkerBackendCapabilities InspectBackend(const WorkerBackend &backend,
                                         u32 requested_width);
bool ExecuteBackendPartitions(const WorkerBackend &backend,
                              const Partition *partitions, u32 partition_count,
                              WorkerTask task, WorkerStats *out_stats);
Result ValidatePlan(const Plan &plan);
const char *ValidateCommonBackendCapabilities(
    const WorkerBackendCapabilities &capabilities);
Result ExecuteSerial(const Plan &plan, Telemetry telemetry);

} // namespace rund::kernel::dispatch::detail
