#include "local.hpp"

namespace workspace_weighted_contract {

int Source() {
  rund::kernel::Workspace missing_workspace{};
  const rund::kernel::ScheduleCompileRequest missing_request{
      .packet_count = 4u,
      .execution_width = 2u,
      .intent = rund::kernel::PartitionIntent::StaticWidth,
      .placement = rund::kernel::PlacementPolicy::WeightedStable,
      .allocation = rund::kernel::AllocationPolicy::AllowGrowth,
  };
  const rund::kernel::PartitionBuild missing_build =
      Compile(missing_workspace, missing_request);
  TEST_ASSERT(!missing_build.ok);
  TEST_ASSERT(std::string_view{missing_build.reason} ==
              "invalid_packet_work_units");

  rund::kernel::Workspace partial_workspace{};
  const std::vector<rund::kernel::u64> partial_units{7u, 8u};
  rund::kernel::ScheduleCompileRequest partial_request = missing_request;
  partial_request.packet_work_units = std::span<const rund::kernel::u64>(
      partial_units.data(), partial_units.size());
  const rund::kernel::PartitionBuild partial_build =
      Compile(partial_workspace, partial_request);
  TEST_ASSERT(!partial_build.ok);
  TEST_ASSERT(std::string_view{partial_build.reason} ==
              "invalid_packet_work_units");

  rund::kernel::Workspace hinted_workspace{};
  const std::vector<rund::kernel::u64> partial_hint_units{100u, 200u};
  const std::vector<rund::kernel::PacketPlacementHint> exact_hints{
      rund::kernel::PacketPlacementHint{.work_units = 4u},
      rund::kernel::PacketPlacementHint{.work_units = 3u},
      rund::kernel::PacketPlacementHint{.work_units = 2u},
      rund::kernel::PacketPlacementHint{.work_units = 1u},
  };
  const rund::kernel::ScheduleCompileRequest hinted_request{
      .packet_count = 4u,
      .execution_width = 2u,
      .intent = rund::kernel::PartitionIntent::StaticWidth,
      .placement = rund::kernel::PlacementPolicy::WeightedStable,
      .allocation = rund::kernel::AllocationPolicy::NoGrowth,
      .packet_work_units = std::span<const rund::kernel::u64>(
          partial_hint_units.data(), partial_hint_units.size()),
      .packet_hints = std::span<const rund::kernel::PacketPlacementHint>(
          exact_hints.data(), exact_hints.size()),
  };
  TEST_ASSERT(Reserve(hinted_workspace, hinted_request) == 0);
  const rund::kernel::PartitionBuild hinted_build =
      Compile(hinted_workspace, hinted_request);
  TEST_ASSERT(hinted_build.ok);
  const std::span<const rund::kernel::u32> hinted_order =
      rund::kernel::ViewOrderedPacketIndices(hinted_workspace);
  TEST_ASSERT(hinted_order[0u] == 0u);
  TEST_ASSERT(rund::kernel::ViewPacketWorkUnits(hinted_workspace)[0u] == 4u);
  TEST_ASSERT(rund::kernel::ViewPacketWorkUnits(hinted_workspace)[1u] == 3u);

  rund::kernel::Workspace oversized_workspace{};
  const std::vector<rund::kernel::u64> oversized_units{1u, 2u, 3u, 4u, 5u};
  rund::kernel::ScheduleCompileRequest oversized_request = hinted_request;
  oversized_request.packet_work_units = std::span<const rund::kernel::u64>(
      oversized_units.data(), oversized_units.size());
  TEST_ASSERT(Reserve(oversized_workspace, oversized_request) == 0);
  const rund::kernel::PartitionBuild oversized_build =
      Compile(oversized_workspace, oversized_request);
  TEST_ASSERT(!oversized_build.ok);
  TEST_ASSERT(std::string_view{oversized_build.reason} ==
              "invalid_packet_work_units");

  rund::kernel::Workspace missing_capacity_workspace{};
  missing_capacity_workspace.schedule.partitions.reserve(2u);
  missing_capacity_workspace.ordered_packet_indices.reserve(4u);
  missing_capacity_workspace.packet_partition_indices.reserve(4u);
  missing_capacity_workspace.ordered_packet_scratch.reserve(4u);
  missing_capacity_workspace.partition_loads.reserve(2u);
  missing_capacity_workspace.partition_counts.reserve(2u);
  missing_capacity_workspace.partition_offsets.reserve(2u);
  missing_capacity_workspace.partition_write_offsets.reserve(2u);
  const std::vector<rund::kernel::u64> zero_units{0u, 2u, 0u, 1u};
  const rund::kernel::ScheduleCompileRequest zero_request = Request(
      4u, 2u, rund::kernel::AllocationPolicy::NoGrowth,
      std::span<const rund::kernel::u64>(zero_units.data(), zero_units.size()));
  const rund::kernel::PartitionBuild missing_capacity_build =
      Compile(missing_capacity_workspace, zero_request);
  TEST_ASSERT(!missing_capacity_build.ok);
  TEST_ASSERT(std::string_view{missing_capacity_build.reason} ==
              "weighted_workspace_capacity_exceeded");
  return 0;
}

} // namespace workspace_weighted_contract
