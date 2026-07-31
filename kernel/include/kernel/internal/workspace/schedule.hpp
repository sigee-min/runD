#pragma once

#include <kernel/schedule/workspace/state.hpp>

namespace rund::kernel::internal {

PartitionBuild CompileWorkspaceSchedule(Workspace& workspace,
                                        const ScheduleCompileRequest& request);

} // namespace rund::kernel::internal
