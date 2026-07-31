#include "../local.hpp"

#include <algorithm>

namespace rund::kernel {
namespace {

bool CanUseAlignedPartitions(const PartitionRequest &request) {
  const u32 width =
      schedule::planner::ClampKernelWidth(request.execution_width);
  const u32 group_count = schedule::planner::ResolveAlignmentGroupCount(
      request.packet_count, request.preferred_alignment_packets);
  return request.preferred_alignment_packets > 1u &&
         request.packet_count % request.preferred_alignment_packets == 0u &&
         group_count > 1u && group_count >= width;
}

PartitionProjection ProjectionFailure(const PartitionRequest &request,
                                      const char *reason,
                                      const u32 alignment_packets = 1u) {
  return PartitionProjection{
      .ok = false,
      .reason = reason,
      .packet_count = request.packet_count,
      .execution_width = request.execution_width,
      .intent = request.intent,
      .alignment_packets = alignment_packets,
  };
}

} // namespace

PartitionProjection ProjectPartitions(const PartitionRequest &request) {
  if (request.packet_count == 0u) {
    return ProjectionFailure(request, "invalid_packet_count");
  }
  if (request.execution_width == 0u) {
    return ProjectionFailure(request, "invalid_execution_width");
  }
  const u32 alignment_packets = CanUseAlignedPartitions(request)
                                    ? request.preferred_alignment_packets
                                    : 1u;
  const u32 partition_units =
      schedule::planner::ResolvePartitionUnits(request, alignment_packets);
  if (partition_units == 0u) {
    return ProjectionFailure(request, "invalid_partition_units",
                             alignment_packets);
  }
  const u32 partition_count =
      schedule::planner::ResolvePartitionCount(request, partition_units);
  const u32 execution_width =
      schedule::planner::ClampKernelWidth(request.execution_width);
  const u32 useful_width = std::min<u32>(execution_width, partition_count);
  return PartitionProjection{
      .ok = true,
      .reason = "pass",
      .packet_count = request.packet_count,
      .execution_width = execution_width,
      .intent = request.intent,
      .placement = PlacementPolicy::Uniform,
      .partition_count = partition_count,
      .alignment_packets = alignment_packets,
      .worker_slot_count = useful_width,
      .fold_slot_count = partition_count,
      .useful_width = useful_width,
  };
}

PartitionProjection ProjectSchedule(const ScheduleCompileRequest &request) {
  PartitionProjection projection =
      ProjectPartitions(ToPartitionRequest(request));
  projection.placement = request.placement;
  return projection;
}

} // namespace rund::kernel
