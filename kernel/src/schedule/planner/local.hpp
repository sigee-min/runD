#pragma once

#include <kernel/schedule/planner/build.hpp>
#include <math32/math32.hpp>

namespace rund::kernel::schedule::planner {

u32 ClampKernelWidth(u32 execution_width);
u32 ResolveAlignmentGroupCount(u32 packet_count, u32 alignment_packets);
u32 ResolvePartitionUnits(const PartitionRequest &request,
                          u32 alignment_packets);
u32 ResolvePartitionCount(const PartitionRequest &request, u32 partition_units);
bool ValidPacketWorkUnits(std::span<const u64> packet_work_units,
                          const PartitionRequest &request);
PartitionBuild FailPartitionBuild(const char *reason);
PartitionBuild
FailProjectedPartitionBuild(const PartitionProjection &projection,
                            const char *reason);

} // namespace rund::kernel::schedule::planner
