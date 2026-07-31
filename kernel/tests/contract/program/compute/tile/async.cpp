#include "contract/support/backend/capabilities.hpp"
#include "contract/support/backend/execute.hpp"
#include "test/assert.hpp"

#include <kernel/program/compute/tile/run.hpp>

#include <array>
#include <atomic>
#include <string_view>
#include <thread>

namespace program_compute_contract {
namespace {

using rund::kernel::ComputeTile;
using rund::kernel::ComputeTileCallbackResult;
using rund::kernel::ComputeTileExecutor;
using rund::kernel::Partition;
using rund::kernel::u32;
using rund::kernel::WorkerBackend;
using rund::kernel::WorkerBackendCapabilities;
using rund::kernel::WorkerStats;
using rund::kernel::WorkerSubmission;
using rund::kernel::WorkerTask;

constexpr u32 kWorkers = 4u;
constexpr u32 kTileUnits = 8u;
constexpr u32 kCount = 65u;
constexpr u32 kTiles = 9u;

struct Immediate final {
  kernel_contract_test::FakePool pool =
      kernel_contract_test::BuildStaticPool(kWorkers);
};

WorkerBackendCapabilities Capabilities(void *const raw) {
  WorkerBackendCapabilities caps = kernel_contract_test::FakeCapabilities(raw);
  caps.supports_async_partitions = true;
  return caps;
}

bool Submit(void *const raw, const Partition *const partitions,
            const u32 partition_count, const WorkerTask task,
            WorkerStats *const stats,
            WorkerSubmission *const submission) noexcept {
  if (raw == nullptr || partitions == nullptr || !task ||
      submission == nullptr || submission->completion.invoke == nullptr ||
      partition_count != kWorkers) {
    return false;
  }
  std::array<std::thread, kWorkers> threads{};
  u32 started = 0u;
  try {
    for (; started < partition_count; ++started) {
      threads[started] = std::thread{
          [&, started] { task.invoke(task.context, partitions[started]); }};
    }
  } catch (...) {
    for (u32 index = 0u; index < started; ++index) {
      threads[index].join();
    }
    return false;
  }
  for (std::thread &thread : threads) {
    thread.join();
  }
  (void)stats;
  submission->completion.invoke(submission->completion.context, true);
  return true;
}

[[nodiscard]] WorkerBackend Backend(Immediate &state) noexcept {
  return WorkerBackend{
      .context = &state.pool,
      .worker_count = kernel_contract_test::FakeWorkerCount,
      .affinity_policy = kernel_contract_test::FakeAffinityPolicy,
      .capabilities = Capabilities,
      .is_nested = kernel_contract_test::FakeIsNested,
      .execute_partitions = kernel_contract_test::FakeExecutePartitions,
      .submit_partitions = Submit,
  };
}

void Ready(void *const raw) noexcept {
  auto *const count = static_cast<std::atomic<u32> *>(raw);
  count->fetch_add(1u, std::memory_order_relaxed);
}

ComputeTileCallbackResult Visit(const void *const raw,
                                const ComputeTile &tile) {
  auto *const visits = const_cast<std::array<u32, kCount> *>(
      static_cast<const std::array<u32, kCount> *>(raw));
  for (u32 index = tile.begin; index < tile.end; ++index) {
    ++(*visits)[index];
  }
  return {};
}

struct Failing final {
  std::array<u32, kCount> *visits = nullptr;
  u32 lower = 0u;
  u32 upper = 0u;
  std::atomic<u32> arrived{0u};
};

ComputeTileCallbackResult Fail(const void *const raw, const ComputeTile &tile) {
  auto *const state = const_cast<Failing *>(static_cast<const Failing *>(raw));
  for (u32 index = tile.begin; index < tile.end; ++index) {
    ++(*state->visits)[index];
  }
  if (tile.index == state->lower) {
    state->arrived.fetch_add(1u, std::memory_order_acq_rel);
    while (state->arrived.load(std::memory_order_acquire) != 2u) {
      std::this_thread::yield();
    }
    return {.ok = false, .reason = "test_async_lower_failure"};
  }
  if (tile.index == state->upper) {
    state->arrived.fetch_add(1u, std::memory_order_acq_rel);
    while (state->arrived.load(std::memory_order_acquire) != 2u) {
      std::this_thread::yield();
    }
    return {.ok = false, .reason = "test_async_upper_failure"};
  }
  return {};
}

int test_async_reuses_sync_projection_without_stale_failure_state() {
  Immediate immediate{};
  const WorkerBackend backend = Backend(immediate);
  ComputeTileExecutor plan{
      backend, kWorkers,
      rund::kernel::physical_tiles(2u, kTileUnits, kTileUnits)};
  const auto prepared = plan.prepare(kCount);
  TEST_ASSERT(prepared.ok);
  TEST_ASSERT(prepared.tile_count == kTiles);
  ComputeTileExecutor run = plan.make_run();
  TEST_ASSERT(run.prepared());

  std::atomic<u32> ready{0u};
  std::array<u32, kCount> failed_visits{};
  Failing failure{
      .visits = &failed_visits,
      .lower = 1u,
      .upper = 3u,
  };
  const auto submitted =
      run.submit_with_erased(backend, &failure, Fail, &ready, Ready);
  TEST_ASSERT(submitted.ok);
  TEST_ASSERT(ready.load(std::memory_order_relaxed) == 1u);
  const auto failed = run.finish();
  TEST_ASSERT(!failed.ok);
  TEST_ASSERT(failed.first_failure_tile == 1u);
  TEST_ASSERT(std::string_view{failed.reason} == "test_async_lower_failure");
  TEST_ASSERT(failed.completed_tile_count >= 3u);
  TEST_ASSERT(failed.completed_tile_count <= kTiles);
  TEST_ASSERT(failed.last_tile_units == 1u);

  std::array<u32, kCount> visits{};
  TEST_ASSERT(
      run.submit_with_erased(backend, &visits, Visit, &ready, Ready).ok);
  TEST_ASSERT(ready.load(std::memory_order_relaxed) == 2u);
  const auto passed = run.finish();
  TEST_ASSERT(passed.ok);
  TEST_ASSERT(passed.first_failure_tile == rund::kernel::kNoComputeTileFailure);
  TEST_ASSERT(passed.completed_tile_count == kTiles);
  TEST_ASSERT(passed.worker_tile_count == kTiles);
  TEST_ASSERT(passed.last_tile_units == 1u);
  for (const u32 count : visits) {
    TEST_ASSERT(count == 1u);
  }
  TEST_ASSERT(std::string_view{run.finish().reason} ==
              "compute_tile_not_ready");
  return 0;
}

} // namespace

int RunComputeTileAsyncContract() {
  return test_async_reuses_sync_projection_without_stale_failure_state();
}

} // namespace program_compute_contract
