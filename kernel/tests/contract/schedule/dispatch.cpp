#include "cases.hpp"

#include "contract/support.hpp"
#include "test/assert.hpp"

#include <kernel/internal/dispatch/kernel.hpp>
#include <kernel/internal/workspace/schedule.hpp>
#include <kernel/dispatch/orchestrator.hpp>
#include <kernel/schedule/workspace.hpp>

#include <array>
#include <atomic>

int RunProgramTileDispatchContract() {
  using namespace kernel_contract_test;

  std::array<std::atomic<rund::kernel::u32>, kMaxVisitedPackets> visits{};
  std::array<rund::kernel::u32, kMaxVisitedPackets> observed_packets{};
  std::atomic<rund::kernel::u32> observed_count{0u};
  VisitContext context{
      .visits = &visits,
      .observed_packets = &observed_packets,
      .observed_count = &observed_count,
      .packet_count = 16u,
  };

  ResetVisits(visits);
  FakePool pool = BuildStaticPool(4u);
  rund::kernel::Workspace workspace{};
  const rund::kernel::PartitionBuild build =
      rund::kernel::internal::CompileWorkspaceSchedule(
          workspace,
          rund::kernel::ScheduleCompileRequest{
              .packet_count = 16u,
              .execution_width = 4u,
              .intent = rund::kernel::PartitionIntent::StaticWidth,
              .preferred_alignment_packets = 1u,
          });
  TEST_ASSERT(build.ok);
  const rund::kernel::ScheduleView schedule = rund::kernel::ViewSchedule(workspace);
  TEST_ASSERT(schedule.partition_count == 4u);
  const rund::kernel::internal::SchedulePlan plan{
      .schedule = schedule,
      .worker_backend = MakeFakeBackend(&pool),
      .context = &context,
      .dispatch = MarkVisitedPackets,
      .collect_worker_stats = true,
  };
  const rund::kernel::Result result = rund::kernel::internal::ExecuteSchedule(plan);
  TEST_ASSERT(result.ok);
  TEST_ASSERT(result.telemetry.useful_width == 4u);
  TEST_ASSERT(result.telemetry.partition_count == schedule.partition_count);
  TEST_ASSERT(result.telemetry.fold_slot_count == schedule.partition_count);
  TEST_ASSERT(result.telemetry.worker_slot_count == 4u);
  TEST_ASSERT(result.telemetry.static_tile_map_used);
  TEST_ASSERT(result.telemetry.global_claim_sync_elided);
  TEST_ASSERT(result.telemetry.claim_fetch_count == 0u);
  TEST_ASSERT(result.telemetry.max_partition_size > 0u);
  TEST_ASSERT(result.telemetry.partition_size_imbalance_milli == 0u);
  TEST_ASSERT(result.telemetry.total_partitions_executed == schedule.partition_count);
  TEST_ASSERT(EachPacketVisitedOnce(visits, 16u));
  return 0;
}
