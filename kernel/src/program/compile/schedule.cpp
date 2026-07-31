#include "local.hpp"
#include "../timing/local.hpp"

#include <kernel/internal/workspace/schedule.hpp>

namespace rund::kernel::program_detail {

PartitionBuild CompileProgramSchedule(Workspace& workspace,
                                      const ScheduleCompileRequest& schedule_request) {
  const TimePoint schedule_compile_start = Now();
  const PartitionBuild schedule_build =
      rund::kernel::internal::CompileWorkspaceSchedule(workspace, schedule_request);
  RecordScheduleCompileCost(workspace, schedule_compile_start);
  return schedule_build;
}

} // namespace rund::kernel::program_detail
