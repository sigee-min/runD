#include "local.hpp"

namespace workspace_weighted_contract {

int Ties() {
  rund::kernel::Workspace tie_workspace{};
  const std::vector<rund::kernel::u64> tie_units{1u, 1u, 1u, 1u, 1u, 1u};
  const rund::kernel::ScheduleCompileRequest tie_request = Request(
      6u, 3u, rund::kernel::AllocationPolicy::AllowGrowth,
      std::span<const rund::kernel::u64>(tie_units.data(), tie_units.size()));
  const rund::kernel::PartitionBuild tie_build =
      Compile(tie_workspace, tie_request);
  TEST_ASSERT(tie_build.ok);
  TEST_ASSERT(ExpectOrder(tie_workspace, {0u, 3u, 1u, 4u, 2u, 5u}) == 0);

  rund::kernel::Workspace eight_tie_workspace{};
  const std::vector<rund::kernel::u64> eight_tie_units(16u, 1u);
  const rund::kernel::ScheduleCompileRequest eight_tie_request =
      Request(16u, 8u, rund::kernel::AllocationPolicy::NoGrowth,
              std::span<const rund::kernel::u64>(eight_tie_units.data(),
                                                 eight_tie_units.size()));
  TEST_ASSERT(Reserve(eight_tie_workspace, eight_tie_request) == 0);
  const rund::kernel::PartitionBuild eight_tie_build =
      Compile(eight_tie_workspace, eight_tie_request);
  TEST_ASSERT(eight_tie_build.ok);
  TEST_ASSERT(
      ExpectOrder(eight_tie_workspace, {0u, 8u, 1u, 9u, 2u, 10u, 3u, 11u, 4u,
                                        12u, 5u, 13u, 6u, 14u, 7u, 15u}) == 0);

  rund::kernel::Workspace eight_weighted_workspace{};
  const std::vector<rund::kernel::u64> eight_weighted_units{9u, 8u, 7u, 6u, 5u,
                                                            4u, 3u, 2u, 1u, 1u};
  const rund::kernel::ScheduleCompileRequest eight_weighted_request =
      Request(static_cast<rund::kernel::u32>(eight_weighted_units.size()), 8u,
              rund::kernel::AllocationPolicy::NoGrowth,
              std::span<const rund::kernel::u64>(eight_weighted_units.data(),
                                                 eight_weighted_units.size()));
  TEST_ASSERT(Reserve(eight_weighted_workspace, eight_weighted_request) == 0);
  const rund::kernel::PartitionBuild eight_weighted_build =
      Compile(eight_weighted_workspace, eight_weighted_request);
  TEST_ASSERT(eight_weighted_build.ok);
  TEST_ASSERT(ExpectOrder(eight_weighted_workspace,
                          {0u, 1u, 2u, 3u, 4u, 5u, 6u, 9u, 7u, 8u}) == 0);
  return 0;
}

} // namespace workspace_weighted_contract
