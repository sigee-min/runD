#include "cases.hpp"

#include "test/assert.hpp"

#include <kernel/internal/workspace/schedule.hpp>
#include <kernel/schedule/workspace.hpp>

#include <span>
#include <vector>

int RunWorkspaceBalanceContract() {
  rund::kernel::Workspace workspace{};
  const std::vector<rund::kernel::u64> work_units{1u, 1u, 10u, 1u};
  const rund::kernel::ScheduleCompileRequest balanced_request{
      .packet_count = 4u,
      .execution_width = 2u,
      .intent = rund::kernel::PartitionIntent::StaticWidth,
      .placement = rund::kernel::PlacementPolicy::ContiguousBalanced,
      .allocation = rund::kernel::AllocationPolicy::NoGrowth,
      .packet_work_units = std::span<const rund::kernel::u64>(work_units.data(), work_units.size()),
  };
  TEST_ASSERT(rund::kernel::ReserveWorkspace(workspace, rund::kernel::ScheduleWorkspaceReservation(balanced_request)));
  const rund::kernel::PartitionBuild build =
      rund::kernel::internal::CompileWorkspaceSchedule(workspace, balanced_request);
  TEST_ASSERT(build.ok);
  TEST_ASSERT(build.no_allocation);
  TEST_ASSERT(build.placement == rund::kernel::PlacementPolicy::ContiguousBalanced);
  TEST_ASSERT(workspace.schedule.partitions.size() == 2u);
  TEST_ASSERT(workspace.schedule.partitions[0u].begin == 0u);
  TEST_ASSERT(workspace.schedule.partitions[0u].end == 3u);
  TEST_ASSERT(workspace.schedule.partitions[1u].begin == 3u);
  TEST_ASSERT(workspace.schedule.partitions[1u].end == 4u);
  return 0;
}
