#pragma once

#include <kernel/dispatch/worker/enums.hpp>

namespace rund::kernel {

struct KernelProgramPlacementMetadata {
  bool has_packet_hints = false;
  bool has_packet_work_units = false;
  u32 alignment_group_packets = 1u;
  u32 locality_bucket_crossing_count = 0u;
  bool has_worker_capacity = false;
  bool worker_capacity_truth = false;
  u32 worker_capacity_imbalance_milli = 0u;
  WorkerTruthLevel affinity_truth_level = WorkerTruthLevel::Unknown;
  bool affinity_hint_only = false;
  const char* affinity_placement_reason = "not_evaluated";
  u64 max_partition_work_units = 0u;
  u64 min_partition_work_units = 0u;
  u32 work_imbalance_milli = 0u;
};

} // namespace rund::kernel
