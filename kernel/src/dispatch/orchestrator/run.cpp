#include "local.hpp"
#include "physical/tile.hpp"

#include <kernel/internal/dispatch/kernel.hpp>

namespace rund::kernel {
namespace {

RunResult FailStatsCapacity(const orchestrator_detail::ProgramPartitionExecution& request,
                            const char* const reason) {
  Result kernel{
      .ok = false,
      .failure_reason = reason,
      .telemetry = Telemetry{
          .packet_count = request.schedule.packet_count,
          .execution_width = request.schedule.execution_width,
          .useful_width = request.schedule.useful_width,
          .partition_count = request.schedule.partition_count,
          .worker_slot_count = request.schedule.worker_slot_count,
          .fold_slot_count = request.schedule.fold_slot_count,
          .placement = request.schedule.placement,
          .no_allocation = request.schedule.no_allocation,
      },
  };
  return RunResult{
      .ok = false,
      .reason = kernel.failure_reason,
      .domain_failed = false,
      .kernel = kernel,
  };
}

} // namespace

RunResult RunPreparedProgram(const RunPreparedProgramRequest& request) {
  if (request.workspace == nullptr) {
    return RunResult{
        .ok = false,
        .reason = "workspace_missing",
    };
  }
  const KernelProgram& program = request.workspace->program;
  if (!program.ok) {
    return RunResult{
        .ok = false,
        .reason = program.reason,
    };
  }
  if (program.schedule.partition_count < request.minimum_partition_count) {
    return RunResult{
        .ok = false,
        .reason = "partition_count_below_min",
    };
  }
  orchestrator_detail::PhysicalTileDispatchAdapter tile_adapter{};
  orchestrator_detail::ProgramPartitionExecution execution{
      .schedule = program.schedule,
      .worker_backend = request.worker_backend,
      .context = request.context,
      .dispatch = request.dispatch,
      .workspace = request.workspace,
      .failure_signal = request.failure_signal,
      .collect_worker_stats = request.collect_worker_stats,
      .worker_stats_sink = request.worker_stats_sink,
      .worker_start_offset_ns_sink = request.worker_start_offset_ns_sink,
      .worker_elapsed_ns_sink = request.worker_elapsed_ns_sink,
      .worker_tail_wait_ns_sink = request.worker_tail_wait_ns_sink,
      .require_no_allocation = request.require_no_allocation,
  };
  if (orchestrator_detail::UseStripedPhysicalTiles(program)) {
    tile_adapter = orchestrator_detail::PhysicalTileDispatchAdapter{
        .context = request.context,
        .dispatch = request.dispatch,
        .failure_signal = request.failure_signal,
        .packet_count = program.schedule.packet_count,
        .execution_width = program.schedule.execution_width,
        .physical_tile_units = program.exec_kernel.physical_tile_units,
        .physical_tile_count = program.exec_kernel.physical_tile_count,
    };
    execution.context = &tile_adapter;
    execution.dispatch = orchestrator_detail::InvokeStripedPhysicalTiles;
  }
  RunResult out = orchestrator_detail::ExecuteProgramPartitions(
      orchestrator_detail::ProgramPartitionExecution{
          .schedule = execution.schedule,
          .worker_backend = execution.worker_backend,
          .context = execution.context,
          .dispatch = execution.dispatch,
          .workspace = execution.workspace,
          .failure_signal = execution.failure_signal,
          .collect_worker_stats = execution.collect_worker_stats,
          .worker_stats_sink = execution.worker_stats_sink,
          .worker_start_offset_ns_sink = execution.worker_start_offset_ns_sink,
          .worker_elapsed_ns_sink = execution.worker_elapsed_ns_sink,
          .worker_tail_wait_ns_sink = execution.worker_tail_wait_ns_sink,
          .require_no_allocation = execution.require_no_allocation,
      });
  RecordTelemetry(*request.workspace, out.kernel);
  orchestrator_detail::AttachPhysicalTileTelemetry(*request.workspace);
  out.kernel.telemetry = request.workspace->telemetry;
  return out;
}

namespace orchestrator_detail {

bool WorkspaceHasWorkerStatsCapacity(const Workspace& workspace,
                                     const u32 worker_count) {
  return GetWorkspaceCapacity(workspace).worker_stats_capacity >= worker_count;
}

void ResizeWorkspaceWorkerStats(Workspace& workspace,
                                const u32 worker_count) {
  workspace.worker_stats_partitions_per_worker.resize(worker_count);
  workspace.worker_stats_start_offset_ns.resize(worker_count);
  workspace.worker_stats_elapsed_ns.resize(worker_count);
  workspace.worker_stats_tail_wait_ns.resize(worker_count);
}

std::span<u32> WorkspaceWorkerPartitionStats(Workspace& workspace) {
  return std::span<u32>(workspace.worker_stats_partitions_per_worker.data(),
                        workspace.worker_stats_partitions_per_worker.size());
}

std::span<u64> WorkspaceWorkerStartOffsetStats(Workspace& workspace) {
  return std::span<u64>(workspace.worker_stats_start_offset_ns.data(),
                        workspace.worker_stats_start_offset_ns.size());
}

std::span<u64> WorkspaceWorkerElapsedStats(Workspace& workspace) {
  return std::span<u64>(workspace.worker_stats_elapsed_ns.data(),
                        workspace.worker_stats_elapsed_ns.size());
}

std::span<u64> WorkspaceWorkerTailWaitStats(Workspace& workspace) {
  return std::span<u64>(workspace.worker_stats_tail_wait_ns.data(),
                        workspace.worker_stats_tail_wait_ns.size());
}

RunResult ExecuteProgramPartitions(const ProgramPartitionExecution& request) {
  std::span<u32> worker_stats_sink = request.worker_stats_sink;
  std::span<u64> worker_start_offset_ns_sink = request.worker_start_offset_ns_sink;
  std::span<u64> worker_elapsed_ns_sink = request.worker_elapsed_ns_sink;
  std::span<u64> worker_tail_wait_ns_sink = request.worker_tail_wait_ns_sink;
  if (worker_stats_sink.empty() &&
      request.require_no_allocation &&
      request.collect_worker_stats &&
      request.workspace != nullptr) {
    if (!WorkspaceHasWorkerStatsCapacity(*request.workspace,
                                         request.schedule.execution_width)) {
      return FailStatsCapacity(request, "worker_stats_capacity_exceeded");
    }
    ResizeWorkspaceWorkerStats(*request.workspace, request.schedule.execution_width);
    worker_stats_sink = WorkspaceWorkerPartitionStats(*request.workspace);
    worker_start_offset_ns_sink = WorkspaceWorkerStartOffsetStats(*request.workspace);
    worker_elapsed_ns_sink = WorkspaceWorkerElapsedStats(*request.workspace);
    worker_tail_wait_ns_sink = WorkspaceWorkerTailWaitStats(*request.workspace);
  }
  const internal::SchedulePlan kernel_plan{
      .schedule = request.schedule,
      .worker_backend = request.worker_backend,
      .context = request.context,
      .dispatch = request.dispatch,
      .collect_worker_stats = request.collect_worker_stats,
      .worker_stats_sink = worker_stats_sink,
      .worker_start_offset_ns_sink = worker_start_offset_ns_sink,
      .worker_elapsed_ns_sink = worker_elapsed_ns_sink,
      .worker_tail_wait_ns_sink = worker_tail_wait_ns_sink,
      .require_no_allocation = request.require_no_allocation,
  };
  RunResult out{};
  out.kernel = internal::ExecuteSchedule(kernel_plan);
  out.domain_failed = request.failure_signal != nullptr &&
                      HasFailure(*request.failure_signal);
  out.ok = out.kernel.ok && !out.domain_failed;
  if (!out.kernel.ok) {
    out.reason = out.kernel.failure_reason;
  } else if (out.domain_failed) {
    const char* const reason = request.failure_signal->reason.load(std::memory_order_acquire);
    out.reason = reason != nullptr ? reason : "dispatch_failed";
  } else {
    out.reason = "pass";
  }
  return out;
}

} // namespace orchestrator_detail

} // namespace rund::kernel
