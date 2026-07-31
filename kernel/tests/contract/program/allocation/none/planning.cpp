#include "contract/program/allocation/none/cases.hpp"
#include "contract/program/allocation/none/request.hpp"

#include "test/assert.hpp"

#include <kernel/program/build.hpp>
#include <kernel/schedule/workspace.hpp>

int RunProgramNoAllocationPlanningContract() {
  rund::kernel::Workspace planning_workspace{};
  rund::kernel::KernelProgramCompileRequest planning_request{
      .schedule = kernel_contract_test::program_no_allocation::ScheduleRequest(),
      .require_no_allocation = true,
      .collect_worker_stats = false,
      .require_dispatch_backend = false,
  };
  TEST_ASSERT(rund::kernel::ReserveWorkspace(planning_workspace,
                                       rund::kernel::KernelProgramWorkspaceReservation(planning_request)));
  const rund::kernel::KernelProgramBuild planning_build =
      rund::kernel::CompileKernelProgram(planning_workspace, planning_request);
  TEST_ASSERT(planning_build.ok);
  TEST_ASSERT(!planning_build.program.dispatch.require_dispatch_backend);
  TEST_ASSERT(planning_build.program.backend_capabilities.backend_width == 0u);
  return 0;
}
