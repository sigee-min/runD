#include "cases.hpp"

#include "test/assert.hpp"

#include <kernel/internal/workspace/schedule.hpp>
#include <kernel/schedule/workspace.hpp>

#include <string_view>

int RunWorkspaceCapacityContract() {
  rund::kernel::Workspace workspace{};
  const rund::kernel::ScheduleCompileRequest no_alloc_request{
      .packet_count = 32u,
      .execution_width = 4u,
      .intent = rund::kernel::PartitionIntent::StaticWidth,
      .placement = rund::kernel::PlacementPolicy::Uniform,
      .allocation = rund::kernel::AllocationPolicy::NoGrowth,
      .preferred_alignment_packets = 8u,
      .locality_bucket_packets = 8u,
  };
  const rund::kernel::PartitionBuild missing_capacity =
      rund::kernel::internal::CompileWorkspaceSchedule(workspace, no_alloc_request);
  TEST_ASSERT(!missing_capacity.ok);
  TEST_ASSERT(std::string_view{missing_capacity.reason} == "schedule_partition_capacity_exceeded");

  TEST_ASSERT(rund::kernel::ReserveWorkspace(workspace, rund::kernel::ScheduleWorkspaceReservation(no_alloc_request)));
  const rund::kernel::WorkspaceCapacity capacity = rund::kernel::GetWorkspaceCapacity(workspace);
  TEST_ASSERT(capacity.schedule_partition_capacity >= 4u);
  const rund::kernel::PartitionBuild build =
      rund::kernel::internal::CompileWorkspaceSchedule(workspace, no_alloc_request);
  TEST_ASSERT(build.ok);
  TEST_ASSERT(build.no_allocation);
  TEST_ASSERT(build.alignment_packets == 8u);
  TEST_ASSERT(workspace.schedule.no_allocation);
  TEST_ASSERT(rund::kernel::ViewSchedule(workspace).fold_slot_count == build.partition_count);
  TEST_ASSERT(rund::kernel::WorkspaceSatisfiesSchedule(workspace, no_alloc_request));
  TEST_ASSERT(workspace.telemetry.compile_cost_measured);
  TEST_ASSERT(workspace.telemetry.capacity_checked);
  TEST_ASSERT(workspace.telemetry.capacity_satisfied);
  TEST_ASSERT(workspace.telemetry.required_schedule_partition_capacity >= build.partition_count);

  // Reset is a warm reuse boundary. It must clear the live schedule without
  // discarding cold-reserved partition storage.
  const auto *const narrow_storage = workspace.schedule.partitions.data();
  const std::size_t narrow_capacity = workspace.schedule.partitions.capacity();
  rund::kernel::ResetWorkspace(workspace);
  TEST_ASSERT(workspace.schedule.partitions.empty());
  TEST_ASSERT(workspace.schedule.partitions.data() == narrow_storage);
  TEST_ASSERT(workspace.schedule.partitions.capacity() == narrow_capacity);
  TEST_ASSERT(workspace.schedule.execution_width == 1u);
  TEST_ASSERT(!workspace.program.ok);
  return 0;
}
