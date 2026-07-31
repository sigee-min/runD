#pragma once

#include <kernel/schedule/workspace/state.hpp>

#include <span>

namespace rund::kernel::program_detail {

[[nodiscard]] inline std::span<const u32> ReportPartitionStats(
    const Workspace& workspace) noexcept {
  if (!workspace.telemetry.partitions_per_worker.empty()) {
    return std::span<const u32>(workspace.telemetry.partitions_per_worker.data(),
                                workspace.telemetry.partitions_per_worker.size());
  }
  return std::span<const u32>(workspace.worker_stats_partitions_per_worker.data(),
                              workspace.worker_stats_partitions_per_worker.size());
}

[[nodiscard]] inline std::span<const u64> ReportStartOffsetStats(
    const Workspace& workspace) noexcept {
  if (!workspace.telemetry.worker_start_offset_ns.empty()) {
    return std::span<const u64>(workspace.telemetry.worker_start_offset_ns.data(),
                                workspace.telemetry.worker_start_offset_ns.size());
  }
  return std::span<const u64>(workspace.worker_stats_start_offset_ns.data(),
                              workspace.worker_stats_start_offset_ns.size());
}

[[nodiscard]] inline std::span<const u64> ReportElapsedStats(
    const Workspace& workspace) noexcept {
  if (!workspace.telemetry.worker_elapsed_ns.empty()) {
    return std::span<const u64>(workspace.telemetry.worker_elapsed_ns.data(),
                                workspace.telemetry.worker_elapsed_ns.size());
  }
  return std::span<const u64>(workspace.worker_stats_elapsed_ns.data(),
                              workspace.worker_stats_elapsed_ns.size());
}

[[nodiscard]] inline std::span<const u64> ReportTailWaitStats(
    const Workspace& workspace) noexcept {
  if (!workspace.telemetry.worker_tail_wait_ns.empty()) {
    return std::span<const u64>(workspace.telemetry.worker_tail_wait_ns.data(),
                                workspace.telemetry.worker_tail_wait_ns.size());
  }
  return std::span<const u64>(workspace.worker_stats_tail_wait_ns.data(),
                              workspace.worker_stats_tail_wait_ns.size());
}

} // namespace rund::kernel::program_detail
