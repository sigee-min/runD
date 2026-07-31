#pragma once

#include <kernel/dispatch/orchestrator.hpp>
#include <kernel/schedule/workspace/api.hpp>

namespace rund::kernel::orchestrator_detail {

struct ProgramPartitionExecution {
  ScheduleView schedule{};
  WorkerBackend worker_backend{};
  void* context = nullptr;
  DispatchFn dispatch = nullptr;
  Workspace* workspace = nullptr;
  FailureSignal* failure_signal = nullptr;
  bool collect_worker_stats = true;
  std::span<u32> worker_stats_sink{};
  std::span<u64> worker_start_offset_ns_sink{};
  std::span<u64> worker_elapsed_ns_sink{};
  std::span<u64> worker_tail_wait_ns_sink{};
  bool require_no_allocation = false;
};

RunResult ExecuteProgramPartitions(const ProgramPartitionExecution& request);
bool WorkspaceHasWorkerStatsCapacity(const Workspace& workspace, u32 worker_count);
void ResizeWorkspaceWorkerStats(Workspace& workspace, u32 worker_count);
std::span<u32> WorkspaceWorkerPartitionStats(Workspace& workspace);
std::span<u64> WorkspaceWorkerStartOffsetStats(Workspace& workspace);
std::span<u64> WorkspaceWorkerElapsedStats(Workspace& workspace);
std::span<u64> WorkspaceWorkerTailWaitStats(Workspace& workspace);

} // namespace rund::kernel::orchestrator_detail
