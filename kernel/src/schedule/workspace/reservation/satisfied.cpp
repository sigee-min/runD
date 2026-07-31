#include "../local.hpp"

namespace rund::kernel::workspace_detail {

bool ReservationSatisfiedByCapacity(const WorkspaceCapacity& capacity,
                                    const WorkspaceReservation& reservation) {
  const WorkspaceReservation required = NormalizeReservation(reservation);
  return capacity.schedule_partition_capacity >= required.schedule_partition_capacity &&
         capacity.packet_work_unit_capacity >= required.packet_work_unit_capacity &&
         capacity.ordered_packet_capacity >= required.ordered_packet_capacity &&
         capacity.packet_partition_capacity >= required.packet_partition_capacity &&
         capacity.ordered_packet_scratch_capacity >= required.ordered_packet_scratch_capacity &&
         capacity.partition_load_capacity >= required.partition_load_capacity &&
         capacity.partition_count_capacity >= required.partition_count_capacity &&
         capacity.partition_offset_capacity >= required.partition_offset_capacity &&
         capacity.partition_write_offset_capacity >= required.partition_write_offset_capacity &&
         capacity.fold_slot_capacity >= required.fold_slot_capacity &&
         capacity.fold_graph_node_capacity >= required.fold_graph_node_capacity &&
         capacity.fold_graph_edge_capacity >= required.fold_graph_edge_capacity &&
         capacity.worker_stats_capacity >= required.worker_stats_capacity;
}

} // namespace rund::kernel::workspace_detail
