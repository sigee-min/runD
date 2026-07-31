#pragma once

#include <kernel/program/tile.hpp>

#include <span>

namespace rund::kernel {

enum class KernelExecutionReportKind : u32 {
  None = 0u,
  Narrow = 1u,
};

struct KernelExecutionReport {
  bool observed = false;
  bool ok = false;
  const char *reason = "not_run";
  KernelExecutionReportKind kind = KernelExecutionReportKind::None;

  u64 packet_count = 0u;
  u32 execution_width = 0u;
  u32 useful_width = 0u;
  u32 partition_count = 0u;
  u32 worker_count = 0u;
  u32 participating_workers = 0u;
  u32 worker_tile_count = 0u;
  u32 min_tiles_per_worker = 0u;
  u32 max_tiles_per_worker = 0u;
  u32 tile_imbalance_milli = 0u;

  bool physical_tiling_enabled = false;
  u64 physical_tile_units = 0u;
  u64 physical_tile_count = 0u;
  PhysicalTileAssignment physical_tile_assignment =
      PhysicalTileAssignment::None;

  bool dispatch_cost_measured = false;
  u64 dispatch_cost_ns = 0u;
  bool backend_execution_cost_measured = false;
  u64 backend_execution_cost_ns = 0u;
  bool telemetry_update_cost_measured = false;
  u64 telemetry_update_cost_ns = 0u;

  bool dispatch_submit_cost_measured = false;
  u64 dispatch_submit_cost_ns = 0u;
  bool dispatch_worker_wake_measured = false;
  u64 dispatch_wake_to_first_worker_ns = 0u;
  u64 dispatch_wake_to_last_worker_ns = 0u;
  bool dispatch_join_wait_measured = false;
  u64 dispatch_join_wait_ns = 0u;

  bool worker_timing_measured = false;
  u64 worker_elapsed_min_ns = 0u;
  u64 worker_elapsed_max_ns = 0u;
  u32 worker_elapsed_imbalance_milli = 0u;
  bool worker_tail_attribution_measured = false;
  u32 slowest_worker_index = 0u;
  u32 slowest_worker_partitions = 0u;
  u64 slowest_worker_elapsed_ns = 0u;
  u64 root_worker_elapsed_ns = 0u;
  u64 root_worker_tail_wait_ns = 0u;

  std::span<const u32> partitions_per_worker{};
  std::span<const u64> worker_start_offset_ns{};
  std::span<const u64> worker_elapsed_ns{};
  std::span<const u64> worker_tail_wait_ns{};
};

[[nodiscard]] inline KernelExecutionReport
invalid_execution_report(const char *const reason) noexcept {
  return KernelExecutionReport{
      .reason = reason == nullptr ? "not_run" : reason,
  };
}

} // namespace rund::kernel
