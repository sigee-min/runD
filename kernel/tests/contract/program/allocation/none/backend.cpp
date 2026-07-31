#include "contract/program/allocation/none/cases.hpp"
#include "contract/program/allocation/none/request.hpp"

#include "contract/support.hpp"
#include "test/assert.hpp"

#include <kernel/program/build.hpp>
#include <kernel/schedule/workspace.hpp>

#include <string_view>

int RunProgramNoAllocationBackendStatsContract() {
  using namespace kernel_contract_test;

  rund::kernel::Workspace workspace{};
  FakePool unsupported_pool = BuildStaticPool(3u);
  unsupported_pool.supports_no_alloc_stats = false;
  rund::kernel::KernelProgramCompileRequest request{
      .schedule = program_no_allocation::ScheduleRequest(),
      .worker_backend = MakeFakeBackend(&unsupported_pool),
      .require_no_allocation = true,
      .collect_worker_stats = true,
  };
  TEST_ASSERT(rund::kernel::ReserveWorkspace(
      workspace, rund::kernel::KernelProgramWorkspaceReservation(request)));
  const rund::kernel::KernelProgramBuild unsupported =
      rund::kernel::CompileKernelProgram(workspace, request);
  TEST_ASSERT(!unsupported.ok);
  TEST_ASSERT(std::string_view{unsupported.reason} ==
              "worker_stats_would_allocate");
  return 0;
}
