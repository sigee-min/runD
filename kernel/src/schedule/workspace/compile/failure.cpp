#include "local.hpp"

namespace rund::kernel::workspace_compile_detail {

PartitionBuild BuildProjectedFailure(Workspace& workspace,
                                     const ScheduleCompileRequest& request,
                                     const PartitionProjection& projection,
                                     const char* const reason) {
  workspace.last_failure_reason = reason;
  return PartitionBuild{
      .ok = false,
      .reason = reason,
      .partition_count = projection.partition_count,
      .alignment_packets = projection.alignment_packets,
      .worker_slot_count = projection.worker_slot_count,
      .fold_slot_count = projection.fold_slot_count,
      .useful_width = projection.useful_width,
      .placement = projection.placement,
      .no_allocation = request.allocation == AllocationPolicy::NoGrowth,
  };
}

} // namespace rund::kernel::workspace_compile_detail
