#pragma once

#include <kernel/schedule/planner/build.hpp>

#include <vector>

namespace rund::kernel {

struct ScheduleView {
  u32 packet_count = 0;
  u32 execution_width = 1;
  PartitionIntent intent = PartitionIntent::StaticWidth;
  PlacementPolicy placement = PlacementPolicy::Uniform;
  u32 alignment_packets = 1;
  u32 packets_per_partition_max = 0;
  u32 worker_slot_count = 0;
  u32 fold_slot_count = 0;
  u32 useful_width = 0;
  bool no_allocation = false;
  const Partition* partitions = nullptr;
  u32 partition_count = 0;
  const u32* ordered_packet_indices = nullptr;
  u32 ordered_packet_count = 0;
};

// Low-level builders live under kernel/internal/schedule/builder.hpp for
// kernel implementation and focused primitive tests only.
ScheduleView ViewSchedule(const Schedule& schedule);
} // namespace rund::kernel
