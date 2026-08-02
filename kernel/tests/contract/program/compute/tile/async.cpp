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
using rund::kernel::ComputeTileRunStorage;
using rund::kernel::ComputeTileRunStorageView;
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
constexpr u32 kBoundedCount = 34u;
constexpr u32 kBoundedTiles = 5u;

struct ExternalRunStorage final {
  ComputeTileRunStorage state{};
  std::array<const char *, kTiles> failures{};
  std::array<u32, kWorkers> worker_tiles{};
  std::array<u32, kWorkers> worker_stats_partitions{};
  std::array<rund::kernel::u64, kWorkers> worker_stats_start_offset_ns{};
  std::array<rund::kernel::u64, kWorkers> worker_stats_elapsed_ns{};
  std::array<rund::kernel::u64, kWorkers> worker_stats_tail_wait_ns{};

  [[nodiscard]] ComputeTileRunStorageView view() noexcept {
    return ComputeTileRunStorageView{
        .state = &state,
        .failure_slots = failures,
        .worker_tiles = worker_tiles,
        .worker_stats_partitions = worker_stats_partitions,
        .worker_stats_start_offset_ns = worker_stats_start_offset_ns,
        .worker_stats_elapsed_ns = worker_stats_elapsed_ns,
        .worker_stats_tail_wait_ns = worker_stats_tail_wait_ns,
    };
  }
};

struct Immediate final {
  kernel_contract_test::FakePool pool =
      kernel_contract_test::BuildStaticPool(kWorkers);
};

struct Deferred final {
  kernel_contract_test::FakePool pool =
      kernel_contract_test::BuildStaticPool(kWorkers);
  const Partition *partitions = nullptr;
  u32 partition_count = 0u;
  WorkerTask task{};
  WorkerSubmission *submission = nullptr;
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

u32 DeferredWorkers(void *const raw) {
  auto *const state = static_cast<Deferred *>(raw);
  return kernel_contract_test::FakeWorkerCount(&state->pool);
}

rund::kernel::WorkerAffinityPolicy DeferredAffinity(void *const raw) {
  auto *const state = static_cast<Deferred *>(raw);
  return kernel_contract_test::FakeAffinityPolicy(&state->pool);
}

WorkerBackendCapabilities DeferredCapabilities(void *const raw) {
  auto *const state = static_cast<Deferred *>(raw);
  WorkerBackendCapabilities caps =
      kernel_contract_test::FakeCapabilities(&state->pool);
  caps.supports_async_partitions = true;
  return caps;
}

bool DeferredNested(void *const raw) {
  auto *const state = static_cast<Deferred *>(raw);
  return kernel_contract_test::FakeIsNested(&state->pool);
}

bool DeferredExecute(void *const raw, const Partition *const partitions,
                     const u32 partition_count, const WorkerTask task,
                     WorkerStats *const stats) {
  auto *const state = static_cast<Deferred *>(raw);
  return kernel_contract_test::FakeExecutePartitions(
      &state->pool, partitions, partition_count, task, stats);
}

bool SubmitDeferred(void *const raw, const Partition *const partitions,
                    const u32 partition_count, const WorkerTask task,
                    WorkerStats *const,
                    WorkerSubmission *const submission) noexcept {
  auto *const state = static_cast<Deferred *>(raw);
  if (state == nullptr || partitions == nullptr || !task ||
      submission == nullptr || submission->completion.invoke == nullptr ||
      state->submission != nullptr) {
    return false;
  }
  state->partitions = partitions;
  state->partition_count = partition_count;
  state->task = task;
  state->submission = submission;
  return true;
}

[[nodiscard]] WorkerBackend Backend(Deferred &state) noexcept {
  return WorkerBackend{
      .context = &state,
      .worker_count = DeferredWorkers,
      .affinity_policy = DeferredAffinity,
      .capabilities = DeferredCapabilities,
      .is_nested = DeferredNested,
      .execute_partitions = DeferredExecute,
      .submit_partitions = SubmitDeferred,
  };
}

void Complete(Deferred &state) noexcept {
  for (u32 index = 0u; index < state.partition_count; ++index) {
    state.task.invoke(state.task.context, state.partitions[index]);
  }
  WorkerSubmission *const submission = state.submission;
  state.partitions = nullptr;
  state.partition_count = 0u;
  state.task = {};
  state.submission = nullptr;
  submission->completion.invoke(submission->completion.context, true);
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

int test_ready_run_blocks_rebind_until_finish_and_invalidates_old_view() {
  Deferred deferred{};
  const WorkerBackend backend = Backend(deferred);
  ComputeTileExecutor source{
      backend, kWorkers,
      rund::kernel::physical_tiles(2u, kTileUnits, kTileUnits)};
  TEST_ASSERT(source.prepare(kCount).ok);
  const auto plan = source.run_plan();
  ExternalRunStorage external{};
  ComputeTileExecutor run = plan.bind(external.view(), kBoundedCount);
  TEST_ASSERT(run.prepared());
  TEST_ASSERT(run.count() == kBoundedCount);
  TEST_ASSERT(run.tile_count() == kBoundedTiles);

  std::atomic<u32> ready{0u};
  std::array<u32, kCount> visits{};
  TEST_ASSERT(
      run.submit_with_erased(backend, &visits, Visit, &ready, Ready).ok);
  TEST_ASSERT(ready.load(std::memory_order_relaxed) == 0u);
  TEST_ASSERT(external.state.busy());

  ComputeTileExecutor inflight_rejected = plan.bind(external.view());
  const auto inflight_busy = inflight_rejected.run(
      [](const ComputeTile &) { return ComputeTileCallbackResult{}; });
  TEST_ASSERT(std::string_view{inflight_busy.reason} ==
              "compute_tile_run_busy");

  Complete(deferred);
  TEST_ASSERT(ready.load(std::memory_order_relaxed) == 1u);
  TEST_ASSERT(external.state.busy());

  ComputeTileExecutor rejected = plan.bind(external.view());
  TEST_ASSERT(!rejected.prepared());
  const auto busy = rejected.run(
      [](const ComputeTile &) { return ComputeTileCallbackResult{}; });
  TEST_ASSERT(std::string_view{busy.reason} == "compute_tile_run_busy");
  const auto bounded = run.finish();
  TEST_ASSERT(bounded.ok);
  TEST_ASSERT(bounded.completed_tile_count == kBoundedTiles);
  TEST_ASSERT(bounded.worker_tile_count == kBoundedTiles);
  TEST_ASSERT(bounded.last_tile_units == 2u);
  for (u32 index = 0u; index < kBoundedCount; ++index) {
    TEST_ASSERT(visits[index] == 1u);
  }
  for (u32 index = kBoundedCount; index < kCount; ++index) {
    TEST_ASSERT(visits[index] == 0u);
  }
  TEST_ASSERT(!external.state.busy());

  ComputeTileExecutor rebound = plan.bind(external.view());
  TEST_ASSERT(rebound.prepared());
  TEST_ASSERT(!run.prepared());
  const auto stale =
      run.run([](const ComputeTile &) { return ComputeTileCallbackResult{}; });
  TEST_ASSERT(std::string_view{stale.reason} == "compute_tile_run_rebound");
  return 0;
}

} // namespace

int RunComputeTileAsyncContract() {
  if (test_async_reuses_sync_projection_without_stale_failure_state() != 0) {
    return 1;
  }
  return test_ready_run_blocks_rebind_until_finish_and_invalidates_old_view();
}

} // namespace program_compute_contract
