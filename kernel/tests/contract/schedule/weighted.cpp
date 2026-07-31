#include "cases.hpp"

#include "test/assert.hpp"

#include <kernel/internal/schedule/builder.hpp>

#include <span>
#include <string_view>
#include <vector>

int RunProgramTileWeightedContract() {
  {
    std::vector<rund::kernel::Partition> partitions{};
    const std::vector<rund::kernel::u64> work_units{1u, 1u, 10u, 1u};
    const rund::kernel::PartitionBuild build =
        rund::kernel::internal::BuildBalancedPartitions(partitions,
                                                  std::span<const rund::kernel::u64>(work_units.data(),
                                                                              work_units.size()),
                                                  rund::kernel::PartitionRequest{
                                                      .packet_count = 4u,
                                                      .execution_width = 2u,
                                                      .intent = rund::kernel::PartitionIntent::StaticWidth,
                                                  });
    TEST_ASSERT(build.ok);
    TEST_ASSERT(partitions.size() == 2u);
    TEST_ASSERT(partitions[0u].begin == 0u);
    TEST_ASSERT(partitions[0u].end == 3u);
    TEST_ASSERT(partitions[1u].begin == 3u);
    TEST_ASSERT(partitions[1u].end == 4u);
  }

  rund::kernel::Schedule schedule{};
  const rund::kernel::ScheduleCompileRequest weighted_request{
      .packet_count = 4u,
      .execution_width = 2u,
      .intent = rund::kernel::PartitionIntent::StaticWidth,
      .placement = rund::kernel::PlacementPolicy::WeightedStable,
      .allocation = rund::kernel::AllocationPolicy::AllowGrowth,
  };
  const rund::kernel::PartitionBuild weighted_build =
      rund::kernel::internal::CompileSchedule(schedule, weighted_request);
  TEST_ASSERT(!weighted_build.ok);
  TEST_ASSERT(std::string_view(weighted_build.reason) == "weighted_schedule_requires_workspace");
  return 0;
}
