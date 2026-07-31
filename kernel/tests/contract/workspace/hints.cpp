#include "cases.hpp"

#include "test/assert.hpp"

#include <kernel/internal/workspace/schedule.hpp>
#include <kernel/schedule/workspace.hpp>

#include <span>
#include <string_view>
#include <vector>

int RunWorkspaceHintedDispatchContract() {
  rund::kernel::Workspace workspace{};
  const std::vector<rund::kernel::PacketPlacementHint> hints{
      rund::kernel::PacketPlacementHint{.locality_bucket_id = 2u, .cache_line_group = 0u, .work_units = 10u},
      rund::kernel::PacketPlacementHint{.locality_bucket_id = 1u, .cache_line_group = 0u, .work_units = 1u},
      rund::kernel::PacketPlacementHint{.locality_bucket_id = 1u, .cache_line_group = 1u, .work_units = 8u},
      rund::kernel::PacketPlacementHint{.locality_bucket_id = 2u, .cache_line_group = 1u, .work_units = 1u},
  };
  const rund::kernel::ScheduleCompileRequest request{
      .packet_count = 4u,
      .execution_width = 2u,
      .intent = rund::kernel::PartitionIntent::StaticWidth,
      .placement = rund::kernel::PlacementPolicy::WeightedStable,
      .allocation = rund::kernel::AllocationPolicy::NoGrowth,
      .alignment_group_packets = 2u,
      .packet_hints = std::span<const rund::kernel::PacketPlacementHint>(hints.data(), hints.size()),
  };
  TEST_ASSERT(rund::kernel::ReserveWorkspace(workspace, rund::kernel::ScheduleWorkspaceReservation(request)));
  const rund::kernel::PartitionBuild hinted_build =
      rund::kernel::internal::CompileWorkspaceSchedule(workspace, request);
  TEST_ASSERT(hinted_build.ok);
  const std::span<const rund::kernel::u32> ordered = rund::kernel::ViewOrderedPacketIndices(workspace);
  TEST_ASSERT(ordered.size() == 4u);
  TEST_ASSERT(ordered[0u] == 2u);
  TEST_ASSERT(ordered[1u] == 3u);
  TEST_ASSERT(ordered[2u] == 1u);
  TEST_ASSERT(ordered[3u] == 0u);
  const rund::kernel::ScheduleView schedule = rund::kernel::ViewSchedule(workspace);
  TEST_ASSERT(schedule.ordered_packet_count == 4u);
  const rund::kernel::KernelProgramPlacementMetadata placement =
      rund::kernel::BuildKernelProgramPlacementMetadata(request,
                                                        schedule,
                                                        rund::kernel::ViewPacketWorkUnits(workspace));
  TEST_ASSERT(placement.has_packet_hints);
  TEST_ASSERT(placement.alignment_group_packets == 2u);
  TEST_ASSERT(placement.max_partition_work_units >= placement.min_partition_work_units);
  TEST_ASSERT(workspace.telemetry.locality_bucket_crossing_count > 0u);
  TEST_ASSERT(workspace.telemetry.locality_bucket_crossing_measured);
  TEST_ASSERT(workspace.program.telemetry_schema.locality_bucket_crossing_measured);
  TEST_ASSERT(workspace.program.ok);
  TEST_ASSERT(std::string_view{workspace.program.reason} == "pass");
  return 0;
}
