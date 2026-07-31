#include "local.hpp"

#include <algorithm>

namespace rund::kernel::workspace_detail {

bool ScatterWeightedPacketOrder(Workspace& workspace,
                                const ScheduleCompileRequest& request,
                                const PartitionProjection& projection) {
  u32 offset = 0u;
  for (u32 partition = 0u; partition < projection.partition_count; ++partition) {
    workspace.partition_offsets[partition] = offset;
    workspace.partition_write_offsets[partition] = offset;
    offset += workspace.partition_counts[partition];
  }
  if (offset != request.packet_count) {
    return false;
  }

  for (const u32 packet_index : workspace.ordered_packet_indices) {
    const u32 partition = workspace.packet_partition_indices[packet_index];
    const u32 write_index = workspace.partition_write_offsets[partition]++;
    workspace.ordered_packet_scratch[write_index] = packet_index;
  }
  std::copy(workspace.ordered_packet_scratch.begin(),
            workspace.ordered_packet_scratch.begin() + request.packet_count,
            workspace.ordered_packet_indices.begin());
  return true;
}

} // namespace rund::kernel::workspace_detail
