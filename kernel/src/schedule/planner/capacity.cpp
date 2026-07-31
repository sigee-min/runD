#include <kernel/internal/schedule/builder.hpp>

#include "capacity/local.hpp"
#include "local.hpp"

#include <algorithm>
#include <cstddef>
#include <span>

namespace rund::kernel::internal {
namespace {

bool ValidCapacities(const PartitionRequest& request, const u32 partition_count) {
  if (request.intent != PartitionIntent::StaticWidth ||
      request.worker_capacity_milli.size() <
          static_cast<std::size_t>(request.execution_width)) {
    return false;
  }
  (void)partition_count;
  for (u32 index = 0u; index < request.execution_width; ++index) {
    if (request.worker_capacity_milli[index] == 0u) {
      return false;
    }
  }
  return true;
}

bool HasPacketWorkUnits(const PartitionRequest& request) {
  return request.packet_work_units.size() == static_cast<std::size_t>(request.packet_count);
}

} // namespace

PartitionBuild BuildCapacityWeightedPartitions(
    std::vector<Partition>& out_partitions,
    const PartitionRequest& request) {
  out_partitions.clear();
  const PartitionProjection projection = ProjectPartitions(request);
  if (!projection.ok) {
    return schedule::planner::FailProjectedPartitionBuild(projection, projection.reason);
  }
  const u32 partition_count = projection.partition_count;
  if (!ValidCapacities(request, partition_count)) {
    return schedule::planner::FailProjectedPartitionBuild(projection, "invalid_worker_capacity");
  }
  if (!request.trust_worker_capacity) {
    return schedule::planner::FailProjectedPartitionBuild(projection, "untrusted_worker_capacity");
  }
  if (!request.packet_work_units.empty() &&
      !schedule::planner::ValidPacketWorkUnits(request.packet_work_units, request)) {
    return schedule::planner::FailProjectedPartitionBuild(projection, "invalid_packet_work_units");
  }
  const u32 alignment = projection.alignment_packets;
  const u32 partition_units = schedule::planner::ResolvePartitionUnits(request, alignment);
  if (out_partitions.capacity() < partition_count) {
    out_partitions.reserve(partition_count);
  }
  if (HasPacketWorkUnits(request)) {
    u32 max_packets = 0u;
    if (!schedule::planner::capacity::BuildMinimaxWorkPartitions(out_partitions,
                                                                 request,
                                                                 partition_count,
                                                                 max_packets)) {
      out_partitions.clear();
      return schedule::planner::FailProjectedPartitionBuild(projection, "capacity_minimax_failed");
    }
    return PartitionBuild{
        .ok = true,
        .reason = "pass",
        .partition_count = static_cast<u32>(out_partitions.size()),
        .alignment_packets = 1u,
        .packets_per_partition_max = max_packets,
        .worker_slot_count = projection.worker_slot_count,
        .fold_slot_count = projection.fold_slot_count,
        .useful_width = projection.useful_width,
        .placement = PlacementPolicy::CapacityWeightedStatic,
    };
  }
  if (!schedule::planner::capacity::BuildMinimaxUnitWidths(out_partitions,
                                                           request,
                                                           partition_count,
                                                           partition_units)) {
    out_partitions.clear();
    return schedule::planner::FailProjectedPartitionBuild(projection, "capacity_minimax_failed");
  }
  u32 max_packets = 0u;
  u32 unit_begin = 0u;
  for (u32 slot = 0u; slot < partition_count; ++slot) {
    const u32 width = out_partitions[slot].end;
    const u32 unit_end = unit_begin + width;
    const u32 begin = unit_begin * alignment;
    const u32 end = unit_end * alignment;
    out_partitions[slot] = Partition{.worker_index = slot, .begin = begin, .end = end};
    max_packets = std::max<u32>(max_packets, end - begin);
    unit_begin = unit_end;
  }
  return PartitionBuild{
      .ok = true,
      .reason = "pass",
      .partition_count = static_cast<u32>(out_partitions.size()),
      .alignment_packets = alignment,
      .packets_per_partition_max = max_packets,
      .worker_slot_count = projection.worker_slot_count,
      .fold_slot_count = projection.fold_slot_count,
      .useful_width = projection.useful_width,
      .placement = PlacementPolicy::CapacityWeightedStatic,
  };
}

} // namespace rund::kernel::internal
