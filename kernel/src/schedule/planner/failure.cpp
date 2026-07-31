#include "local.hpp"

namespace rund::kernel::schedule::planner {

PartitionBuild FailPartitionBuild(const char* const reason) {
  return PartitionBuild{
      .ok = false,
      .reason = reason,
  };
}

PartitionBuild FailProjectedPartitionBuild(const PartitionProjection& projection,
                                           const char* const reason) {
  return PartitionBuild{
      .ok = false,
      .reason = reason,
      .partition_count = projection.partition_count,
      .alignment_packets = projection.alignment_packets,
      .worker_slot_count = projection.worker_slot_count,
      .fold_slot_count = projection.fold_slot_count,
      .useful_width = projection.useful_width,
      .placement = projection.placement,
  };
}

} // namespace rund::kernel::schedule::planner
