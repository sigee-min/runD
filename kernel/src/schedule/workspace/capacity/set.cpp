#include "../local.hpp"

namespace rund::kernel {

KernelProgramCapacitySet ToKernelProgramCapacitySet(const WorkspaceReservation& reservation) {
  const WorkspaceReservation normalized = workspace_detail::NormalizeReservation(reservation);
  return KernelProgramCapacitySet{
      .schedule_partition_capacity = normalized.schedule_partition_capacity,
      .packet_capacity = normalized.packet_capacity,
      .packet_work_unit_capacity = normalized.packet_work_unit_capacity,
      .ordered_packet_capacity = normalized.ordered_packet_capacity,
      .packet_partition_capacity = normalized.packet_partition_capacity,
      .ordered_packet_scratch_capacity = normalized.ordered_packet_scratch_capacity,
      .partition_load_capacity = normalized.partition_load_capacity,
      .partition_count_capacity = normalized.partition_count_capacity,
      .partition_offset_capacity = normalized.partition_offset_capacity,
      .partition_write_offset_capacity = normalized.partition_write_offset_capacity,
      .fold_slot_capacity = normalized.fold_slot_capacity,
      .fold_graph_node_capacity = normalized.fold_graph_node_capacity,
      .fold_graph_edge_capacity = normalized.fold_graph_edge_capacity,
      .worker_stats_capacity = normalized.worker_stats_capacity,
  };
}

KernelProgramCapacitySet ToKernelProgramCapacitySet(const WorkspaceCapacity& capacity) {
  return KernelProgramCapacitySet{
      .schedule_partition_capacity = capacity.schedule_partition_capacity,
      .packet_capacity = capacity.packet_capacity,
      .packet_work_unit_capacity = capacity.packet_work_unit_capacity,
      .ordered_packet_capacity = capacity.ordered_packet_capacity,
      .packet_partition_capacity = capacity.packet_partition_capacity,
      .ordered_packet_scratch_capacity = capacity.ordered_packet_scratch_capacity,
      .partition_load_capacity = capacity.partition_load_capacity,
      .partition_count_capacity = capacity.partition_count_capacity,
      .partition_offset_capacity = capacity.partition_offset_capacity,
      .partition_write_offset_capacity = capacity.partition_write_offset_capacity,
      .fold_slot_capacity = capacity.fold_slot_capacity,
      .fold_graph_node_capacity = capacity.fold_graph_node_capacity,
      .fold_graph_edge_capacity = capacity.fold_graph_edge_capacity,
      .worker_stats_capacity = capacity.worker_stats_capacity,
  };
}

} // namespace rund::kernel
