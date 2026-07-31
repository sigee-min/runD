#include "../local.hpp"

#include <algorithm>
#include <limits>

namespace rund::kernel::workspace_detail {
namespace {

u32 CapacityMargin(const u32 available, const u32 required) {
  return available >= required ? available - required : 0u;
}

u32 MinNonZeroMargin(const u32 current, const u32 available, const u32 required) {
  if (required == 0u) {
    return current;
  }
  const u32 margin = CapacityMargin(available, required);
  return current == std::numeric_limits<u32>::max() ? margin : std::min(current, margin);
}

} // namespace

u32 MinimumCapacityMargin(const WorkspaceReservation& required,
                          const WorkspaceCapacity& available) {
  const WorkspaceReservation normalized = NormalizeReservation(required);
  u32 margin = std::numeric_limits<u32>::max();
  margin = MinNonZeroMargin(margin,
                            available.schedule_partition_capacity,
                            normalized.schedule_partition_capacity);
  margin = MinNonZeroMargin(margin,
                            available.packet_work_unit_capacity,
                            normalized.packet_work_unit_capacity);
  margin = MinNonZeroMargin(margin,
                            available.ordered_packet_capacity,
                            normalized.ordered_packet_capacity);
  margin = MinNonZeroMargin(margin,
                            available.packet_partition_capacity,
                            normalized.packet_partition_capacity);
  margin = MinNonZeroMargin(margin,
                            available.ordered_packet_scratch_capacity,
                            normalized.ordered_packet_scratch_capacity);
  margin = MinNonZeroMargin(margin,
                            available.partition_load_capacity,
                            normalized.partition_load_capacity);
  margin = MinNonZeroMargin(margin,
                            available.partition_count_capacity,
                            normalized.partition_count_capacity);
  margin = MinNonZeroMargin(margin,
                            available.partition_offset_capacity,
                            normalized.partition_offset_capacity);
  margin = MinNonZeroMargin(margin,
                            available.partition_write_offset_capacity,
                            normalized.partition_write_offset_capacity);
  margin = MinNonZeroMargin(margin, available.fold_slot_capacity, normalized.fold_slot_capacity);
  margin = MinNonZeroMargin(margin,
                            available.fold_graph_node_capacity,
                            normalized.fold_graph_node_capacity);
  margin = MinNonZeroMargin(margin,
                            available.fold_graph_edge_capacity,
                            normalized.fold_graph_edge_capacity);
  margin = MinNonZeroMargin(margin,
                            available.worker_stats_capacity,
                            normalized.worker_stats_capacity);
  return margin == std::numeric_limits<u32>::max() ? 0u : margin;
}

} // namespace rund::kernel::workspace_detail
