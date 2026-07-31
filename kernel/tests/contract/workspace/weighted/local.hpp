#pragma once

#include "test/assert.hpp"

#include <kernel/internal/workspace/schedule.hpp>
#include <kernel/schedule/workspace.hpp>

#include <span>
#include <string_view>
#include <vector>

namespace workspace_weighted_contract {

[[nodiscard]] inline rund::kernel::PartitionBuild
Compile(rund::kernel::Workspace &workspace,
        const rund::kernel::ScheduleCompileRequest &request) {
  return rund::kernel::internal::CompileWorkspaceSchedule(workspace, request);
}

[[nodiscard]] inline rund::kernel::ScheduleCompileRequest
Request(rund::kernel::u32 packet_count, rund::kernel::u32 execution_width,
        rund::kernel::AllocationPolicy allocation,
        std::span<const rund::kernel::u64> work_units) {
  return rund::kernel::ScheduleCompileRequest{
      .packet_count = packet_count,
      .execution_width = execution_width,
      .intent = rund::kernel::PartitionIntent::StaticWidth,
      .placement = rund::kernel::PlacementPolicy::WeightedStable,
      .allocation = allocation,
      .packet_work_units = work_units,
  };
}

[[nodiscard]] inline int
Reserve(rund::kernel::Workspace &workspace,
        const rund::kernel::ScheduleCompileRequest &request) {
  TEST_ASSERT(rund::kernel::ReserveWorkspace(
      workspace, rund::kernel::ScheduleWorkspaceReservation(request)));
  return 0;
}

[[nodiscard]] inline int
ExpectOrder(rund::kernel::Workspace &workspace,
            const std::vector<rund::kernel::u32> &expected) {
  const std::span<const rund::kernel::u32> actual =
      rund::kernel::ViewOrderedPacketIndices(workspace);
  TEST_ASSERT(actual.size() == expected.size());
  for (rund::kernel::u32 index = 0u; index < actual.size(); ++index) {
    TEST_ASSERT(actual[index] == expected[index]);
  }
  return 0;
}

[[nodiscard]] inline int
ExpectNoGrowthOrder(const std::vector<rund::kernel::u64> &work_units,
                    const std::vector<rund::kernel::u32> &expected) {
  rund::kernel::Workspace workspace{};
  const rund::kernel::ScheduleCompileRequest request = Request(
      static_cast<rund::kernel::u32>(work_units.size()), 1u,
      rund::kernel::AllocationPolicy::NoGrowth,
      std::span<const rund::kernel::u64>(work_units.data(), work_units.size()));
  TEST_ASSERT(Reserve(workspace, request) == 0);
  const rund::kernel::PartitionBuild build = Compile(workspace, request);
  TEST_ASSERT(build.ok);
  TEST_ASSERT(ExpectOrder(workspace, expected) == 0);
  return 0;
}

int Basic();
int Ties();
int Range();
int Source();
int Alias();

} // namespace workspace_weighted_contract
