#include "../local.hpp"

#include <kernel/reduction/fold/graph/api.hpp>

#include <kernel/internal/program/contract.hpp>

namespace rund::kernel {

WorkspaceReservation ScheduleWorkspaceReservation(const ScheduleCompileRequest& request) {
  const PartitionProjection projection = ProjectSchedule(request);
  if (!projection.ok) {
    return WorkspaceReservation{};
  }
  const u32 partition_capacity = projection.partition_count;
  const bool needs_packet_scratch =
      request.placement == PlacementPolicy::WeightedStable ||
      (request.placement == PlacementPolicy::ContiguousBalanced &&
       request.packet_work_units.empty() &&
       request.packet_hints.size() == static_cast<std::size_t>(request.packet_count));
  const u32 packet_capacity = needs_packet_scratch ? request.packet_count : 0u;
  return workspace_detail::NormalizeReservation(WorkspaceReservation{
      .schedule_partition_capacity = partition_capacity,
      .packet_capacity = packet_capacity,
      .packet_work_unit_capacity = packet_capacity,
      .ordered_packet_capacity = request.placement == PlacementPolicy::WeightedStable ? request.packet_count : 0u,
      .packet_partition_capacity = request.placement == PlacementPolicy::WeightedStable ? request.packet_count : 0u,
      .ordered_packet_scratch_capacity = request.placement == PlacementPolicy::WeightedStable ? request.packet_count : 0u,
      .partition_load_capacity = request.placement == PlacementPolicy::WeightedStable ? partition_capacity : 0u,
      .partition_count_capacity = request.placement == PlacementPolicy::WeightedStable ? partition_capacity : 0u,
      .partition_offset_capacity = request.placement == PlacementPolicy::WeightedStable ? partition_capacity : 0u,
      .partition_write_offset_capacity = request.placement == PlacementPolicy::WeightedStable ? partition_capacity : 0u,
      .fold_slot_capacity = FoldGraphFixedBinaryTreeScratchSlotCount(projection.fold_slot_count),
      .fold_graph_node_capacity = FoldGraphFixedBinaryTreeNodeCount(projection.fold_slot_count),
      .fold_graph_edge_capacity = FoldGraphFixedBinaryTreeEdgeCount(projection.fold_slot_count),
  });
}

WorkspaceReservation KernelProgramWorkspaceReservation(const KernelProgramCompileRequest& request) {
  WorkspaceReservation reservation = ScheduleWorkspaceReservation(request.schedule);
  if (internal::ProgramRequiresNoGrowth(request) && request.collect_worker_stats) {
    reservation.worker_stats_capacity = request.schedule.execution_width;
  }
  return workspace_detail::NormalizeReservation(reservation);
}

bool WorkspaceSatisfiesSchedule(const Workspace& workspace, const ScheduleCompileRequest& request) {
  const PartitionProjection projection = ProjectSchedule(request);
  if (!projection.ok) {
    return false;
  }
  return WorkspaceSatisfiesReservation(workspace, ScheduleWorkspaceReservation(request));
}

} // namespace rund::kernel
