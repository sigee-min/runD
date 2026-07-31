#include <kernel/internal/schedule/builder.hpp>

#include "local.hpp"

namespace rund::kernel::internal {
namespace {

bool HasCapacity(const std::vector<Partition>& out_partitions,
                 const u32 partition_count,
                 const AllocationPolicy allocation) {
  return allocation == AllocationPolicy::AllowGrowth ||
         out_partitions.capacity() >= static_cast<std::size_t>(partition_count);
}

} // namespace

PartitionBuild BuildSchedule(
    Schedule& out_schedule,
    const PartitionRequest& request) {
  return CompileSchedule(out_schedule, ToScheduleCompileRequest(request));
}

PartitionBuild CompileSchedule(
    Schedule& out_schedule,
    const ScheduleCompileRequest& request) {
  out_schedule.packet_count = request.packet_count;
  out_schedule.execution_width = request.execution_width;
  out_schedule.intent = request.intent;
  out_schedule.placement = request.placement;
  out_schedule.alignment_packets = 1u;
  out_schedule.packets_per_partition_max = 0u;
  out_schedule.worker_slot_count = 0u;
  out_schedule.fold_slot_count = 0u;
  out_schedule.useful_width = 0u;
  out_schedule.no_allocation = request.allocation == AllocationPolicy::NoGrowth;

  const PartitionProjection projection = ProjectSchedule(request);
  if (!projection.ok) {
    out_schedule.partitions.clear();
    return schedule::planner::FailProjectedPartitionBuild(projection, projection.reason);
  }
  if (!HasCapacity(out_schedule.partitions, projection.partition_count, request.allocation)) {
    out_schedule.partitions.clear();
    return schedule::planner::FailProjectedPartitionBuild(projection, "schedule_partition_capacity_exceeded");
  }

  PartitionBuild build{};
  const PartitionRequest partition_request = ToPartitionRequest(request);
  switch (request.placement) {
    case PlacementPolicy::Uniform:
      build = BuildPartitions(out_schedule.partitions, partition_request);
      break;
    case PlacementPolicy::ContiguousBalanced:
      build = BuildBalancedPartitions(out_schedule.partitions,
                                      request.packet_work_units,
                                      partition_request);
      break;
    case PlacementPolicy::CapacityWeightedStatic:
      build = BuildCapacityWeightedPartitions(out_schedule.partitions, partition_request);
      break;
    case PlacementPolicy::WeightedStable:
      out_schedule.partitions.clear();
      return schedule::planner::FailProjectedPartitionBuild(projection, "weighted_schedule_requires_workspace");
  }
  if (build.ok) {
    out_schedule.alignment_packets = build.alignment_packets;
    out_schedule.packets_per_partition_max = build.packets_per_partition_max;
    out_schedule.worker_slot_count = build.worker_slot_count;
    out_schedule.fold_slot_count = build.fold_slot_count;
    out_schedule.useful_width = build.useful_width;
    out_schedule.no_allocation = request.allocation == AllocationPolicy::NoGrowth;
  } else {
    out_schedule.partitions.clear();
  }
  build.no_allocation = request.allocation == AllocationPolicy::NoGrowth;
  return build;
}

} // namespace rund::kernel::internal
