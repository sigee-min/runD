#pragma once

#include "../local.hpp"

namespace rund::kernel::workspace_compile_detail {

PartitionBuild BuildProjectedFailure(Workspace& workspace,
                                     const ScheduleCompileRequest& request,
                                     const PartitionProjection& projection,
                                     const char* reason);

PartitionBuild CompileStandardSchedule(Workspace& workspace,
                                       const ScheduleCompileRequest& request,
                                       const PartitionProjection& projection);

PartitionBuild CompileWeightedSchedule(Workspace& workspace,
                                       const ScheduleCompileRequest& request,
                                       const PartitionProjection& projection);

void StoreWeightedScheduleState(Workspace& workspace,
                                const ScheduleCompileRequest& request,
                                const PartitionBuild& build);

} // namespace rund::kernel::workspace_compile_detail
