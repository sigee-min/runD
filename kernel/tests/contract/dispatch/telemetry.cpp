#include "contract/dispatch/cases.hpp"

#include "contract/support.hpp"
#include "test/assert.hpp"

#include <kernel/internal/dispatch/kernel.hpp>

#include <array>
#include <limits>
#include <vector>

int RunDispatchTelemetryContract() {
  const std::array<rund::kernel::Partition, 2u> large_partitions{
      rund::kernel::Partition{.worker_index = 0u, .begin = 0u, .end = 5000000u},
      rund::kernel::Partition{.worker_index = 1u, .begin = 0u, .end = 10000000u},
  };
  const rund::kernel::internal::Plan plan{
      .packet_count = 10000000u,
      .execution_width = 2u,
      .partitions = large_partitions.data(),
      .partition_count = static_cast<rund::kernel::u32>(large_partitions.size()),
  };
  const rund::kernel::Result result = rund::kernel::internal::Execute(plan);
  TEST_ASSERT(!result.ok);
  TEST_ASSERT(result.telemetry.partition_size_imbalance_milli == 333u);

  std::array<rund::kernel::u32, 2u> counts{5000000u, 10000000u};
  rund::kernel::WorkerStats stats{
      .worker_count = 2u,
      .dispatch_submit_cost_ns = 11u,
      .dispatch_submit_cost_measured = true,
      .dispatch_wake_to_first_worker_ns = 13u,
      .dispatch_wake_to_last_worker_ns = 17u,
      .dispatch_worker_wake_measured = true,
      .dispatch_join_wait_ns = 19u,
      .dispatch_join_wait_measured = true,
      .worker_start_skew_ns = 4u,
      .worker_finish_skew_ns = 8u,
      .barrier_wait_ns = 12u,
      .worker_elapsed_min_ns = 100u,
      .worker_elapsed_max_ns = 140u,
      .worker_elapsed_imbalance_milli = 1400u,
      .worker_timing_measured = true,
      .slowest_worker_index = 1u,
      .slowest_worker_partitions = 2u,
      .slowest_worker_elapsed_ns = 140u,
      .root_worker_elapsed_ns = 100u,
      .root_worker_tail_wait_ns = 40u,
      .worker_tail_attribution_measured = true,
      .partitions_per_worker = std::vector<rund::kernel::u32>(counts.begin(), counts.end()),
      .worker_start_offset_ns = std::vector<rund::kernel::u64>{0u, 4u},
      .worker_elapsed_ns = std::vector<rund::kernel::u64>{100u, 140u},
      .worker_tail_wait_ns = std::vector<rund::kernel::u64>{40u, 0u},
  };
  rund::kernel::WorkerBackend backend{
      .context = &stats,
      .worker_count = [](void*) -> rund::kernel::u32 { return 2u; },
      .affinity_policy =
          [](void*) -> rund::kernel::WorkerAffinityPolicy {
            return rund::kernel::WorkerAffinityPolicy::Static;
          },
      .capabilities =
          [](void*) -> rund::kernel::WorkerBackendCapabilities {
            return rund::kernel::WorkerBackendCapabilities{
                .backend_width = 2u,
                .width_matches_request = true,
                .supports_static_partitions = true,
                .supports_static_tile_map = true,
                .supports_claim_free_static_tiles = true,
                .affinity_policy = rund::kernel::WorkerAffinityPolicy::Static,
            };
          },
      .execute_partitions =
          [](void* context,
             const rund::kernel::Partition*,
             rund::kernel::u32,
             rund::kernel::WorkerTask,
             rund::kernel::WorkerStats* out_stats) -> bool {
            if (out_stats == nullptr) {
              return false;
            }
            *out_stats = *static_cast<rund::kernel::WorkerStats*>(context);
            return true;
          },
  };
  const std::array<rund::kernel::Partition, 2u> worker_partitions{
      rund::kernel::Partition{.worker_index = 0u, .begin = 0u, .end = 1u},
      rund::kernel::Partition{.worker_index = 1u, .begin = 1u, .end = 2u},
  };
  const rund::kernel::internal::Plan worker_plan{
      .packet_count = 2u,
      .execution_width = 2u,
      .partitions = worker_partitions.data(),
      .partition_count = static_cast<rund::kernel::u32>(worker_partitions.size()),
      .worker_backend = backend,
      .dispatch = [](void*, const rund::kernel::Partition&) {},
  };
  const rund::kernel::Result worker_result = rund::kernel::internal::Execute(worker_plan);
  TEST_ASSERT(worker_result.ok);
  TEST_ASSERT(worker_result.telemetry.worker_partition_imbalance_milli == 333u);
  TEST_ASSERT(worker_result.telemetry.dispatch_submit_cost_ns == 11u);
  TEST_ASSERT(worker_result.telemetry.dispatch_submit_cost_measured);
  TEST_ASSERT(worker_result.telemetry.dispatch_wake_to_first_worker_ns == 13u);
  TEST_ASSERT(worker_result.telemetry.dispatch_wake_to_last_worker_ns == 17u);
  TEST_ASSERT(worker_result.telemetry.dispatch_worker_wake_measured);
  TEST_ASSERT(worker_result.telemetry.dispatch_join_wait_ns == 19u);
  TEST_ASSERT(worker_result.telemetry.dispatch_join_wait_measured);
  TEST_ASSERT(worker_result.telemetry.worker_start_skew_ns == 4u);
  TEST_ASSERT(worker_result.telemetry.barrier_wait_ns == 12u);
  TEST_ASSERT(worker_result.telemetry.worker_timing_measured);
  TEST_ASSERT(worker_result.telemetry.slowest_worker_index == 1u);
  TEST_ASSERT(worker_result.telemetry.slowest_worker_partitions == 2u);
  TEST_ASSERT(worker_result.telemetry.slowest_worker_elapsed_ns == 140u);
  TEST_ASSERT(worker_result.telemetry.root_worker_elapsed_ns == 100u);
  TEST_ASSERT(worker_result.telemetry.root_worker_tail_wait_ns == 40u);
  TEST_ASSERT(worker_result.telemetry.worker_tail_attribution_measured);
  TEST_ASSERT(worker_result.telemetry.worker_start_offset_ns.size() == 2u);
  TEST_ASSERT(worker_result.telemetry.worker_start_offset_ns[1u] == 4u);
  TEST_ASSERT(worker_result.telemetry.worker_elapsed_ns.size() == 2u);
  TEST_ASSERT(worker_result.telemetry.worker_elapsed_ns[1u] == 140u);
  TEST_ASSERT(worker_result.telemetry.worker_tail_wait_ns.size() == 2u);
  TEST_ASSERT(worker_result.telemetry.worker_tail_wait_ns[0u] == 40u);

  return 0;
}
