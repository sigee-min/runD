#include "../local.hpp"

#include <algorithm>

namespace rund::kernel {

bool ReserveWorkspace(Workspace& workspace, const WorkspaceReservation& reservation) {
  const WorkspaceReservation normalized = workspace_detail::NormalizeReservation(reservation);
  try {
    workspace.schedule.partitions.reserve(normalized.schedule_partition_capacity);
    workspace.packet_work_units.reserve(normalized.packet_work_unit_capacity);
    workspace.ordered_packet_indices.reserve(normalized.ordered_packet_capacity);
    workspace.packet_partition_indices.reserve(normalized.packet_partition_capacity);
    workspace.ordered_packet_scratch.reserve(normalized.ordered_packet_scratch_capacity);
    workspace.partition_loads.reserve(normalized.partition_load_capacity);
    workspace.partition_counts.reserve(normalized.partition_count_capacity);
    workspace.partition_offsets.reserve(normalized.partition_offset_capacity);
    workspace.partition_write_offsets.reserve(normalized.partition_write_offset_capacity);
    workspace.fold_slots.values.reserve(normalized.fold_slot_capacity);
    workspace.fold_graph.partition_fold_slots.reserve(normalized.fold_slot_capacity);
    workspace.fold_graph.nodes.reserve(normalized.fold_graph_node_capacity);
    workspace.fold_graph.reduction_edges.reserve(normalized.fold_graph_edge_capacity);
    workspace.worker_stats_partitions_per_worker.reserve(normalized.worker_stats_capacity);
    workspace.worker_stats_start_offset_ns.reserve(normalized.worker_stats_capacity);
    workspace.worker_stats_elapsed_ns.reserve(normalized.worker_stats_capacity);
    workspace.worker_stats_tail_wait_ns.reserve(normalized.worker_stats_capacity);
  } catch (...) {
    return false;
  }
  return true;
}

WorkspaceCapacity GetWorkspaceCapacity(const Workspace& workspace) {
  return WorkspaceCapacity{
      .schedule_partition_capacity = rund::math32::detail::ScalarSatU32(workspace.schedule.partitions.capacity()),
      .packet_capacity = rund::math32::detail::ScalarSatU32(std::min({
          workspace.packet_work_units.capacity(),
          workspace.ordered_packet_indices.capacity(),
          workspace.packet_partition_indices.capacity(),
          workspace.ordered_packet_scratch.capacity()})),
      .packet_work_unit_capacity = rund::math32::detail::ScalarSatU32(workspace.packet_work_units.capacity()),
      .ordered_packet_capacity = rund::math32::detail::ScalarSatU32(workspace.ordered_packet_indices.capacity()),
      .packet_partition_capacity = rund::math32::detail::ScalarSatU32(workspace.packet_partition_indices.capacity()),
      .ordered_packet_scratch_capacity = rund::math32::detail::ScalarSatU32(workspace.ordered_packet_scratch.capacity()),
      .partition_load_capacity = rund::math32::detail::ScalarSatU32(workspace.partition_loads.capacity()),
      .partition_count_capacity = rund::math32::detail::ScalarSatU32(workspace.partition_counts.capacity()),
      .partition_offset_capacity = rund::math32::detail::ScalarSatU32(workspace.partition_offsets.capacity()),
      .partition_write_offset_capacity = rund::math32::detail::ScalarSatU32(workspace.partition_write_offsets.capacity()),
      .fold_slot_capacity = rund::math32::detail::ScalarSatU32(
          std::min(workspace.fold_slots.values.capacity(),
                   workspace.fold_graph.partition_fold_slots.capacity())),
      .fold_graph_node_capacity = rund::math32::detail::ScalarSatU32(workspace.fold_graph.nodes.capacity()),
      .fold_graph_edge_capacity = rund::math32::detail::ScalarSatU32(workspace.fold_graph.reduction_edges.capacity()),
      .worker_stats_capacity = rund::math32::detail::ScalarSatU32(
          std::min({workspace.worker_stats_partitions_per_worker.capacity(),
                    workspace.worker_stats_start_offset_ns.capacity(),
                    workspace.worker_stats_elapsed_ns.capacity(),
                    workspace.worker_stats_tail_wait_ns.capacity()})),
  };
}

bool WorkspaceSatisfiesReservation(const Workspace& workspace, const WorkspaceReservation& reservation) {
  return workspace_detail::ReservationSatisfiedByCapacity(GetWorkspaceCapacity(workspace), reservation);
}

} // namespace rund::kernel
