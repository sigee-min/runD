#pragma once

#include <kernel/dispatch/kernel.hpp>
#include <kernel/program/model.hpp>
#include <kernel/reduction/fold/graph/state.hpp>
#include <kernel/reduction/fold/slots.hpp>
#include <kernel/schedule/planner/build.hpp>

#include <vector>

namespace rund::kernel {

struct Workspace {
  Schedule schedule{};
  FoldSlots fold_slots{};
  FoldGraph fold_graph{};
  KernelProgram program{};
  u64 program_generation = 0u;
  std::vector<u64> packet_work_units{};
  std::vector<u32> ordered_packet_indices{};
  std::vector<u32> packet_partition_indices{};
  std::vector<u32> ordered_packet_scratch{};
  std::vector<u64> partition_loads{};
  std::vector<u32> partition_counts{};
  std::vector<u32> partition_offsets{};
  std::vector<u32> partition_write_offsets{};
  std::vector<u32> worker_stats_partitions_per_worker{};
  std::vector<u64> worker_stats_start_offset_ns{};
  std::vector<u64> worker_stats_elapsed_ns{};
  std::vector<u64> worker_stats_tail_wait_ns{};
  Telemetry telemetry{};
  const char* last_failure_reason = "not_run";
};

} // namespace rund::kernel
