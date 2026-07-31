#include "local.hpp"

namespace workspace_weighted_contract {

int Basic() {
  rund::kernel::Workspace workspace{};
  const std::vector<rund::kernel::u64> work_units{1u, 10u, 1u, 1u};
  const rund::kernel::ScheduleCompileRequest request = Request(
      4u, 2u, rund::kernel::AllocationPolicy::AllowGrowth,
      std::span<const rund::kernel::u64>(work_units.data(), work_units.size()));
  const rund::kernel::PartitionBuild build = Compile(workspace, request);
  TEST_ASSERT(build.ok);
  TEST_ASSERT(build.placement == rund::kernel::PlacementPolicy::WeightedStable);
  TEST_ASSERT(ExpectOrder(workspace, {1u, 0u, 2u, 3u}) == 0);

  rund::kernel::ScheduleCompileRequest no_alloc = request;
  no_alloc.allocation = rund::kernel::AllocationPolicy::NoGrowth;
  TEST_ASSERT(Reserve(workspace, no_alloc) == 0);
  const rund::kernel::PartitionBuild no_alloc_build =
      Compile(workspace, no_alloc);
  TEST_ASSERT(no_alloc_build.ok);
  TEST_ASSERT(no_alloc_build.no_allocation);
  TEST_ASSERT(no_alloc_build.placement ==
              rund::kernel::PlacementPolicy::WeightedStable);
  TEST_ASSERT(workspace.schedule.no_allocation);
  TEST_ASSERT(workspace.schedule.partitions.size() == 2u);
  TEST_ASSERT(ExpectOrder(workspace, {1u, 0u, 2u, 3u}) == 0);
  return 0;
}

} // namespace workspace_weighted_contract
