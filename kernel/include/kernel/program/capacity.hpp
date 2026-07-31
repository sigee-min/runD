#pragma once

#include <kernel/core/model.hpp>

namespace rund::kernel {

struct KernelProgramCapacitySet {
  u32 schedule_partition_capacity = 0u;
  u32 packet_capacity = 0u;
  u32 packet_work_unit_capacity = 0u;
  u32 ordered_packet_capacity = 0u;
  u32 packet_partition_capacity = 0u;
  u32 ordered_packet_scratch_capacity = 0u;
  u32 partition_load_capacity = 0u;
  u32 partition_count_capacity = 0u;
  u32 partition_offset_capacity = 0u;
  u32 partition_write_offset_capacity = 0u;
  u32 fold_slot_capacity = 0u;
  u32 fold_graph_node_capacity = 0u;
  u32 fold_graph_edge_capacity = 0u;
  u32 worker_stats_capacity = 0u;
};

struct KernelProgramCapacityProof {
  bool checked = false;
  bool satisfied = false;
  const char* reason = "not_checked";
  KernelProgramCapacitySet required{};
  KernelProgramCapacitySet available{};
  u32 no_alloc_capacity_margin = 0u;
};

} // namespace rund::kernel
