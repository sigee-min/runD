#pragma once

#include <kernel/program/report/execution.hpp>
#include <kernel/program/report/reason.hpp>
#include <kernel/program/report/stats.hpp>
#include <kernel/schedule/workspace/state.hpp>

namespace rund::kernel {

[[nodiscard]] inline KernelExecutionReport
execution_report(const Workspace &workspace) noexcept {
  const Telemetry &telemetry = workspace.telemetry;
  const KernelProgramTilePlan &tile_plan = workspace.program.exec_kernel;
  const bool observed = telemetry.packet_count > 0u || workspace.program.ok ||
                        workspace.program_generation > 0u;
  return KernelExecutionReport{
      .observed =
          observed ||
          program_detail::IsObservedReason(workspace.last_failure_reason),
      .ok = program_detail::IsPassReason(workspace.last_failure_reason),
      .reason = workspace.last_failure_reason == nullptr
                    ? "not_run"
                    : workspace.last_failure_reason,
      .kind = observed ? KernelExecutionReportKind::Narrow
                       : KernelExecutionReportKind::None,
      .packet_count = telemetry.packet_count,
      .execution_width = telemetry.execution_width,
      .useful_width = telemetry.useful_width,
      .partition_count = telemetry.partition_count,
      .worker_count = telemetry.worker_count,
      .participating_workers = telemetry.participating_workers,
      .worker_tile_count = telemetry.worker_tile_count,
      .min_tiles_per_worker = telemetry.min_tiles_per_worker,
      .max_tiles_per_worker = telemetry.max_tiles_per_worker,
      .tile_imbalance_milli = telemetry.tile_imbalance_milli,
      .physical_tiling_enabled = tile_plan.physical_tiling_enabled,
      .physical_tile_units = tile_plan.physical_tile_units,
      .physical_tile_count = tile_plan.physical_tile_count,
      .physical_tile_assignment = tile_plan.physical_tile_assignment,
      .dispatch_cost_measured = telemetry.dispatch_cost_measured,
      .dispatch_cost_ns = telemetry.dispatch_cost_ns,
      .backend_execution_cost_measured =
          telemetry.backend_execution_cost_measured,
      .backend_execution_cost_ns = telemetry.backend_execution_cost_ns,
      .telemetry_update_cost_measured =
          telemetry.telemetry_update_cost_measured,
      .telemetry_update_cost_ns = telemetry.telemetry_update_cost_ns,
      .dispatch_submit_cost_measured = telemetry.dispatch_submit_cost_measured,
      .dispatch_submit_cost_ns = telemetry.dispatch_submit_cost_ns,
      .dispatch_worker_wake_measured = telemetry.dispatch_worker_wake_measured,
      .dispatch_wake_to_first_worker_ns =
          telemetry.dispatch_wake_to_first_worker_ns,
      .dispatch_wake_to_last_worker_ns =
          telemetry.dispatch_wake_to_last_worker_ns,
      .dispatch_join_wait_measured = telemetry.dispatch_join_wait_measured,
      .dispatch_join_wait_ns = telemetry.dispatch_join_wait_ns,
      .worker_timing_measured = telemetry.worker_timing_measured,
      .worker_elapsed_min_ns = telemetry.worker_elapsed_min_ns,
      .worker_elapsed_max_ns = telemetry.worker_elapsed_max_ns,
      .worker_elapsed_imbalance_milli =
          telemetry.worker_elapsed_imbalance_milli,
      .worker_tail_attribution_measured =
          telemetry.worker_tail_attribution_measured,
      .slowest_worker_index = telemetry.slowest_worker_index,
      .slowest_worker_partitions = telemetry.slowest_worker_partitions,
      .slowest_worker_elapsed_ns = telemetry.slowest_worker_elapsed_ns,
      .root_worker_elapsed_ns = telemetry.root_worker_elapsed_ns,
      .root_worker_tail_wait_ns = telemetry.root_worker_tail_wait_ns,
      .partitions_per_worker = program_detail::ReportPartitionStats(workspace),
      .worker_start_offset_ns =
          program_detail::ReportStartOffsetStats(workspace),
      .worker_elapsed_ns = program_detail::ReportElapsedStats(workspace),
      .worker_tail_wait_ns = program_detail::ReportTailWaitStats(workspace),
  };
}

} // namespace rund::kernel
