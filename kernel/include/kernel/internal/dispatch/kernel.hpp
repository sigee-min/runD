#pragma once

#include <kernel/dispatch/telemetry.hpp>
#include <kernel/dispatch/worker/backend.hpp>
#include <kernel/schedule/planner/view.hpp>

#include <span>

namespace rund::kernel::internal {

struct Plan {
  u32 packet_count = 0;
  u32 execution_width = 1;
  PlacementPolicy placement = PlacementPolicy::Uniform;
  u32 alignment_packets = 1;
  u32 packets_per_partition_max = 0;
  u32 worker_slot_count = 0;
  u32 fold_slot_count = 0;
  u32 useful_width = 0;
  bool no_allocation = false;
  const Partition *partitions = nullptr;
  u32 partition_count = 0;
  const u32 *ordered_packet_indices = nullptr;
  u32 ordered_packet_count = 0;
  WorkerBackend worker_backend{};
  void *context = nullptr;
  DispatchFn dispatch = nullptr;
  bool collect_worker_stats = true;
  std::span<u32> worker_stats_sink{};
  std::span<u64> worker_start_offset_ns_sink{};
  std::span<u64> worker_elapsed_ns_sink{};
  std::span<u64> worker_tail_wait_ns_sink{};
  bool require_no_allocation = false;
};

struct SchedulePlan {
  ScheduleView schedule{};
  WorkerBackend worker_backend{};
  void *context = nullptr;
  DispatchFn dispatch = nullptr;
  bool collect_worker_stats = true;
  std::span<u32> worker_stats_sink{};
  std::span<u64> worker_start_offset_ns_sink{};
  std::span<u64> worker_elapsed_ns_sink{};
  std::span<u64> worker_tail_wait_ns_sink{};
  bool require_no_allocation = false;
};

Result Execute(const Plan &plan);
Result ExecuteSchedule(const SchedulePlan &plan);

} // namespace rund::kernel::internal
