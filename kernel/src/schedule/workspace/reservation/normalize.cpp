#include "../local.hpp"

#include <kernel/reduction/fold/graph/api.hpp>

#include <algorithm>

namespace rund::kernel::workspace_detail {

WorkspaceReservation NormalizeReservation(WorkspaceReservation reservation) {
  reservation.packet_work_unit_capacity =
      std::max(reservation.packet_work_unit_capacity, reservation.packet_capacity);
  reservation.ordered_packet_capacity =
      std::max(reservation.ordered_packet_capacity, reservation.packet_capacity);
  reservation.packet_partition_capacity =
      std::max(reservation.packet_partition_capacity, reservation.packet_capacity);
  reservation.ordered_packet_scratch_capacity =
      std::max(reservation.ordered_packet_scratch_capacity, reservation.packet_capacity);
  reservation.partition_load_capacity =
      std::max(reservation.partition_load_capacity, reservation.schedule_partition_capacity);
  reservation.partition_count_capacity =
      std::max(reservation.partition_count_capacity, reservation.schedule_partition_capacity);
  reservation.partition_offset_capacity =
      std::max(reservation.partition_offset_capacity, reservation.schedule_partition_capacity);
  reservation.partition_write_offset_capacity =
      std::max(reservation.partition_write_offset_capacity, reservation.schedule_partition_capacity);
  const u32 graph_partition_capacity = reservation.schedule_partition_capacity;
  reservation.fold_slot_capacity =
      std::max(reservation.fold_slot_capacity, graph_partition_capacity);
  reservation.fold_graph_node_capacity =
      std::max(reservation.fold_graph_node_capacity,
               FoldGraphFixedBinaryTreeNodeCount(graph_partition_capacity));
  reservation.fold_graph_edge_capacity =
      std::max(reservation.fold_graph_edge_capacity,
               FoldGraphFixedBinaryTreeEdgeCount(graph_partition_capacity));
  return reservation;
}

} // namespace rund::kernel::workspace_detail
