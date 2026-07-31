#include "assign/eight.hpp"
#include "assign/heap.hpp"
#include "order.hpp"

#include <algorithm>

namespace rund::kernel::workspace_detail {

void AssignWeightedPacketPartitions(
    Workspace &workspace, const ScheduleCompileRequest &request,
    const std::span<const u64> resolved_work_units) {
  BuildWeightedPacketOrder(workspace, request, resolved_work_units);
  std::fill(workspace.partition_loads.begin(), workspace.partition_loads.end(),
            0u);
  std::fill(workspace.partition_counts.begin(),
            workspace.partition_counts.end(), 0u);
  const u32 partition_count =
      static_cast<u32>(workspace.partition_loads.size());
  if (partition_count == 0u) {
    return;
  }
  if (partition_count == 8u) {
    AssignEightPartitions(workspace, resolved_work_units);
    return;
  }
  AssignWeightedPacketPartitionsHeap(workspace, resolved_work_units);
}

} // namespace rund::kernel::workspace_detail
