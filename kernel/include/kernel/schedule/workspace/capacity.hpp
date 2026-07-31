#pragma once

#include <kernel/core/model.hpp>

namespace rund::kernel {

struct WorkspaceReservation {
  u32 schedule_partition_capacity = 0;
  u32 packet_capacity = 0;
  u32 packet_work_unit_capacity = 0;
  u32 ordered_packet_capacity = 0;
  u32 packet_partition_capacity = 0;
  u32 ordered_packet_scratch_capacity = 0;
  u32 partition_load_capacity = 0;
  u32 partition_count_capacity = 0;
  u32 partition_offset_capacity = 0;
  u32 partition_write_offset_capacity = 0;
  u32 fold_slot_capacity = 0;
  u32 fold_graph_node_capacity = 0;
  u32 fold_graph_edge_capacity = 0;
  u32 worker_stats_capacity = 0;
};

struct WorkspaceCapacity {
  u32 schedule_partition_capacity = 0;
  u32 packet_capacity = 0;
  u32 packet_work_unit_capacity = 0;
  u32 ordered_packet_capacity = 0;
  u32 packet_partition_capacity = 0;
  u32 ordered_packet_scratch_capacity = 0;
  u32 partition_load_capacity = 0;
  u32 partition_count_capacity = 0;
  u32 partition_offset_capacity = 0;
  u32 partition_write_offset_capacity = 0;
  u32 fold_slot_capacity = 0;
  u32 fold_graph_node_capacity = 0;
  u32 fold_graph_edge_capacity = 0;
  u32 worker_stats_capacity = 0;
};

} // namespace rund::kernel
