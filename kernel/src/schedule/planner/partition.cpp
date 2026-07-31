#include "local.hpp"

#include <algorithm>

namespace rund::kernel::schedule::planner {
u32 ResolveAlignmentGroupCount(const u32 packet_count,
                               const u32 alignment_packets) {
  if (alignment_packets <= 1u || packet_count % alignment_packets != 0u) {
    return packet_count;
  }
  return packet_count / alignment_packets;
}

u32 ResolvePartitionUnits(const PartitionRequest &request,
                          const u32 alignment_packets) {
  if (alignment_packets <= 1u) {
    return request.packet_count;
  }
  return ResolveAlignmentGroupCount(request.packet_count, alignment_packets);
}

u32 ResolvePartitionCount(const PartitionRequest &request,
                          const u32 partition_units) {
  const u32 width = ClampKernelWidth(request.execution_width);
  return std::max<u32>(1u, std::min<u32>(partition_units, width));
}

} // namespace rund::kernel::schedule::planner
