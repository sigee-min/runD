#include "cases.hpp"

#include "test/assert.hpp"

#include <kernel/internal/schedule/builder.hpp>

#include <span>
#include <string_view>
#include <vector>

int RunProgramTileCapacityContract() {
  {
    std::vector<rund::kernel::Partition> partitions{};
    const std::vector<rund::kernel::u32> capacities{1000u, 500u, 1500u};
    const rund::kernel::PartitionBuild build =
        rund::kernel::internal::BuildCapacityWeightedPartitions(
            partitions,
            rund::kernel::PartitionRequest{
                .packet_count = 300u,
                .execution_width = 3u,
                .intent = rund::kernel::PartitionIntent::StaticWidth,
                .preferred_alignment_packets = 10u,
                .worker_capacity_milli =
                    std::span<const rund::kernel::u32>(capacities.data(), capacities.size()),
                .trust_worker_capacity = true,
            });
    TEST_ASSERT(build.ok);
    TEST_ASSERT(build.placement == rund::kernel::PlacementPolicy::CapacityWeightedStatic);
    TEST_ASSERT(partitions[0u].begin == 0u);
    TEST_ASSERT(partitions[0u].end == 100u);
    TEST_ASSERT(partitions[1u].begin == 100u);
    TEST_ASSERT(partitions[1u].end == 150u);
    TEST_ASSERT(partitions[2u].begin == 150u);
    TEST_ASSERT(partitions[2u].end == 300u);
  }

  {
    std::vector<rund::kernel::Partition> partitions{};
    const rund::kernel::PartitionBuild build =
        rund::kernel::internal::BuildCapacityWeightedPartitions(
            partitions,
            rund::kernel::PartitionRequest{
                .packet_count = 16u,
                .execution_width = 2u,
                .intent = rund::kernel::PartitionIntent::StaticWidth,
            });
    TEST_ASSERT(!build.ok);
    TEST_ASSERT(std::string_view{build.reason} == "invalid_worker_capacity");
  }

  {
    std::vector<rund::kernel::Partition> partitions{};
    const std::vector<rund::kernel::u32> capacities{3u, 5u};
    const rund::kernel::PartitionBuild build =
        rund::kernel::internal::BuildCapacityWeightedPartitions(
            partitions,
            rund::kernel::PartitionRequest{
                .packet_count = 4u,
                .execution_width = 2u,
                .intent = rund::kernel::PartitionIntent::StaticWidth,
                .worker_capacity_milli =
                    std::span<const rund::kernel::u32>(capacities.data(), capacities.size()),
                .trust_worker_capacity = true,
            });
    TEST_ASSERT(build.ok);
    TEST_ASSERT(partitions[0u].begin == 0u);
    TEST_ASSERT(partitions[0u].end == 1u);
    TEST_ASSERT(partitions[1u].begin == 1u);
    TEST_ASSERT(partitions[1u].end == 4u);
  }

  {
    std::vector<rund::kernel::Partition> partitions{};
    const std::vector<rund::kernel::u32> capacities{3u, 5u};
    const rund::kernel::PartitionBuild build =
        rund::kernel::internal::BuildCapacityWeightedPartitions(
            partitions,
            rund::kernel::PartitionRequest{
                .packet_count = 4u,
                .execution_width = 2u,
                .intent = rund::kernel::PartitionIntent::StaticWidth,
                .worker_capacity_milli =
                    std::span<const rund::kernel::u32>(capacities.data(), capacities.size()),
            });
    TEST_ASSERT(!build.ok);
    TEST_ASSERT(std::string_view{build.reason} == "untrusted_worker_capacity");
  }

  {
    std::vector<rund::kernel::Partition> partitions{};
    const std::vector<rund::kernel::u64> work_units{1u, 1u, 1u, 9u, 1u, 1u};
    const std::vector<rund::kernel::u32> capacities{1000u, 2000u};
    const rund::kernel::PartitionBuild build =
        rund::kernel::internal::BuildCapacityWeightedPartitions(
            partitions,
            rund::kernel::PartitionRequest{
                .packet_count = 6u,
                .execution_width = 2u,
                .intent = rund::kernel::PartitionIntent::StaticWidth,
                .packet_work_units =
                    std::span<const rund::kernel::u64>(work_units.data(), work_units.size()),
                .worker_capacity_milli =
                    std::span<const rund::kernel::u32>(capacities.data(), capacities.size()),
                .trust_worker_capacity = true,
            });
    TEST_ASSERT(build.ok);
    TEST_ASSERT(partitions[0u].begin == 0u);
    TEST_ASSERT(partitions[0u].end == 3u);
    TEST_ASSERT(partitions[1u].begin == 3u);
    TEST_ASSERT(partitions[1u].end == 6u);
  }
  return 0;
}
