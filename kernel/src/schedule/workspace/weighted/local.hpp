#pragma once

#include "../local.hpp"

namespace rund::kernel::workspace_detail {

void PrepareWeightedNoGrowthBuffers(Workspace& workspace,
                                    const ScheduleCompileRequest& request,
                                    const PartitionProjection& projection);
void AssignWeightedPacketPartitions(Workspace& workspace,
                                    const ScheduleCompileRequest& request,
                                    std::span<const u64> resolved_work_units);
bool ScatterWeightedPacketOrder(Workspace& workspace,
                                const ScheduleCompileRequest& request,
                                const PartitionProjection& projection);
bool BuildWeightedSchedulePartitions(Workspace& workspace,
                                     const ScheduleCompileRequest& request,
                                     const PartitionProjection& projection,
                                     u32& out_max_packets);
void StoreWeightedNoGrowthScheduleState(Workspace& workspace,
                                        const ScheduleCompileRequest& request,
                                        const PartitionProjection& projection,
                                        u32 max_packets);
PartitionBuild BuildWeightedNoGrowthSuccess(const PartitionProjection& projection, u32 max_packets);

} // namespace rund::kernel::workspace_detail
