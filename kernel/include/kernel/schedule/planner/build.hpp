#pragma once

#include <kernel/schedule/planner/request.hpp>

#include <vector>

namespace rund::kernel {

struct PartitionProjection {
  bool ok = false;
  const char* reason = "not_run";
  u32 packet_count = 0;
  u32 execution_width = 1;
  PartitionIntent intent = PartitionIntent::StaticWidth;
  PlacementPolicy placement = PlacementPolicy::Uniform;
  u32 partition_count = 0;
  u32 alignment_packets = 1;
  u32 worker_slot_count = 0;
  u32 fold_slot_count = 0;
  u32 useful_width = 0;
};

struct PartitionBuild {
  bool ok = false;
  const char* reason = "not_run";
  u32 partition_count = 0;
  u32 alignment_packets = 1;
  u32 packets_per_partition_max = 0;
  u32 worker_slot_count = 0;
  u32 fold_slot_count = 0;
  u32 useful_width = 0;
  PlacementPolicy placement = PlacementPolicy::Uniform;
  bool no_allocation = false;
};

struct Schedule {
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
  std::vector<Partition> partitions{};
};

PartitionProjection ProjectPartitions(const PartitionRequest& request);
PartitionProjection ProjectSchedule(const ScheduleCompileRequest& request);

} // namespace rund::kernel
