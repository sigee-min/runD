#include "local.hpp"

#include <limits>

namespace workspace_weighted_contract {

int Range() {
  rund::kernel::Workspace zero_workspace{};
  const std::vector<rund::kernel::u64> zero_units{0u, 2u, 0u, 1u};
  const rund::kernel::ScheduleCompileRequest zero_request = Request(
      4u, 2u, rund::kernel::AllocationPolicy::NoGrowth,
      std::span<const rund::kernel::u64>(zero_units.data(), zero_units.size()));
  TEST_ASSERT(Reserve(zero_workspace, zero_request) == 0);
  const rund::kernel::PartitionBuild zero_build =
      Compile(zero_workspace, zero_request);
  TEST_ASSERT(zero_build.ok);
  TEST_ASSERT(ExpectOrder(zero_workspace, {1u, 3u, 0u, 2u}) == 0);

  rund::kernel::Workspace threshold_workspace{};
  std::vector<rund::kernel::u64> threshold_units(64u);
  for (rund::kernel::u32 packet = 0u; packet < threshold_units.size();
       ++packet) {
    threshold_units[packet] = packet + 1u;
  }
  const rund::kernel::ScheduleCompileRequest threshold_request =
      Request(static_cast<rund::kernel::u32>(threshold_units.size()), 1u,
              rund::kernel::AllocationPolicy::NoGrowth,
              std::span<const rund::kernel::u64>(threshold_units.data(),
                                                 threshold_units.size()));
  TEST_ASSERT(Reserve(threshold_workspace, threshold_request) == 0);
  const rund::kernel::PartitionBuild threshold_build =
      Compile(threshold_workspace, threshold_request);
  TEST_ASSERT(threshold_build.ok);
  const std::span<const rund::kernel::u32> threshold_order =
      rund::kernel::ViewOrderedPacketIndices(threshold_workspace);
  TEST_ASSERT(threshold_order.size() == threshold_units.size());
  for (rund::kernel::u32 index = 0u; index < threshold_order.size(); ++index) {
    TEST_ASSERT(
        threshold_order[index] ==
        static_cast<rund::kernel::u32>(threshold_order.size() - index - 1u));
  }

  rund::kernel::Workspace wide_workspace{};
  std::vector<rund::kernel::u64> wide_units(65u);
  for (rund::kernel::u32 packet = 0u; packet < wide_units.size(); ++packet) {
    wide_units[packet] = packet + 1u;
  }
  const rund::kernel::ScheduleCompileRequest wide_request = Request(
      static_cast<rund::kernel::u32>(wide_units.size()), 1u,
      rund::kernel::AllocationPolicy::NoGrowth,
      std::span<const rund::kernel::u64>(wide_units.data(), wide_units.size()));
  TEST_ASSERT(Reserve(wide_workspace, wide_request) == 0);
  const rund::kernel::PartitionBuild wide_build =
      Compile(wide_workspace, wide_request);
  TEST_ASSERT(wide_build.ok);
  const std::span<const rund::kernel::u32> wide_order =
      rund::kernel::ViewOrderedPacketIndices(wide_workspace);
  TEST_ASSERT(wide_order.size() == wide_units.size());
  for (rund::kernel::u32 index = 0u; index < wide_order.size(); ++index) {
    TEST_ASSERT(wide_order[index] ==
                static_cast<rund::kernel::u32>(wide_order.size() - index - 1u));
  }

  rund::kernel::Workspace sparse_workspace{};
  const std::vector<rund::kernel::u64> sparse_units{1u, 1000u, 1u, 2u};
  const rund::kernel::ScheduleCompileRequest sparse_request =
      Request(4u, 2u, rund::kernel::AllocationPolicy::NoGrowth,
              std::span<const rund::kernel::u64>(sparse_units.data(),
                                                 sparse_units.size()));
  TEST_ASSERT(Reserve(sparse_workspace, sparse_request) == 0);
  const rund::kernel::PartitionBuild sparse_build =
      Compile(sparse_workspace, sparse_request);
  TEST_ASSERT(sparse_build.ok);
  TEST_ASSERT(ExpectOrder(sparse_workspace, {1u, 3u, 0u, 2u}) == 0);

  TEST_ASSERT(ExpectNoGrowthOrder({0x00FFu, 0x0100u, 0x0001u, 0x0100u},
                                  {1u, 3u, 0u, 2u}) == 0);

  const rund::kernel::u64 max_u64 =
      std::numeric_limits<rund::kernel::u64>::max();
  TEST_ASSERT(ExpectNoGrowthOrder({0u, max_u64, max_u64 - 1u, 1u, max_u64},
                                  {1u, 4u, 2u, 0u, 3u}) == 0);
  return 0;
}

} // namespace workspace_weighted_contract
