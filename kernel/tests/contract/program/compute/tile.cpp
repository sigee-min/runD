#include "contract/program/compute/local.hpp"
#include "contract/support/allocation.hpp"
#include "contract/support/backend/factory.hpp"
#include "test/assert.hpp"

#include <kernel/program/compute/tile/run.hpp>
#include <kernel/program/executor.hpp>

#include <array>
#include <atomic>
#include <cstdio>
#include <limits>
#include <string_view>
#include <utility>

namespace program_compute_contract {
namespace {

using rund::kernel::ComputeTile;
using rund::kernel::ComputeTileCallbackResult;
using rund::kernel::ComputeTileExecutor;
using rund::kernel::ComputeTileRetainedMemory;
using rund::kernel::ComputeTileRunStorage;
using rund::kernel::ComputeTileRunStorageView;
using rund::kernel::PlanComputeTileRunMemory;
using rund::kernel::u32;
using rund::kernel::u64;

constexpr u32 kWorkers = 4u;
constexpr u32 kTileUnits = 32u;
constexpr u32 kTargetTilesPerWorker = 4u;
constexpr u32 kBoundary = kWorkers * kTargetTilesPerWorker * kTileUnits;

template <std::size_t FailureSlots, std::size_t Workers>
struct ExternalRunStorage final {
  ComputeTileRunStorage state{};
  std::array<const char *, FailureSlots> failures{};
  std::array<u32, Workers> worker_tiles{};
  std::array<u32, Workers> worker_stats_partitions{};
  std::array<u64, Workers> worker_stats_start_offset_ns{};
  std::array<u64, Workers> worker_stats_elapsed_ns{};
  std::array<u64, Workers> worker_stats_tail_wait_ns{};

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

[[nodiscard]] constexpr u64 AddBytes(const u64 left, const u64 right) noexcept {
  return right > std::numeric_limits<u64>::max() - left
             ? std::numeric_limits<u64>::max()
             : left + right;
}

[[nodiscard]] u64 TotalBytes(const ComputeTileRetainedMemory memory) noexcept {
  u64 total = AddBytes(memory.state_bytes, memory.workspace_bytes);
  total = AddBytes(total, memory.failure_slot_bytes);
  total = AddBytes(total, memory.worker_tile_bytes);
  return AddBytes(total, memory.async_context_bytes);
}

[[nodiscard]] bool SameMemory(const ComputeTileRetainedMemory left,
                              const ComputeTileRetainedMemory right) noexcept {
  return left.state_bytes == right.state_bytes &&
         left.workspace_bytes == right.workspace_bytes &&
         left.failure_slot_bytes == right.failure_slot_bytes &&
         left.worker_tile_bytes == right.worker_tile_bytes &&
         left.async_context_bytes == right.async_context_bytes &&
         left.total_bytes == right.total_bytes;
}

ComputeTileExecutor MakeExecutor(const rund::kernel::WorkerBackend backend,
                                 const u32 workers) {
  return ComputeTileExecutor{
      backend, workers,
      rund::kernel::physical_tiles(kTargetTilesPerWorker, 1u, kTileUnits)};
}

int CheckBoundary(const u32 count, const u32 expected_tiles,
                  const u32 expected_tail) {
  auto pool = kernel_contract_test::BuildStaticPool(kWorkers);
  ComputeTileExecutor executor =
      MakeExecutor(kernel_contract_test::MakeFakeBackend(&pool), kWorkers);
  const auto prepared = executor.prepare(count);
  if (!prepared.ok) {
    std::fprintf(stderr, "compute tile prepare failed: %s\n", prepared.reason);
  }
  TEST_ASSERT(prepared.ok);
  TEST_ASSERT(prepared.tile_units == kTileUnits);
  TEST_ASSERT(prepared.tile_count == expected_tiles);
  ComputeTileExecutor run_executor = executor.make_run();
  TEST_ASSERT(run_executor.prepared());

  std::array<std::atomic<u32>, kBoundary + 2u> visits{};
  std::atomic<bool> invalid_tile{false};
  const auto run = run_executor.run([&](const ComputeTile &tile) {
    if (tile.begin >= tile.end || tile.end > count) {
      invalid_tile.store(true, std::memory_order_relaxed);
      return ComputeTileCallbackResult{
          .ok = false,
          .reason = "test_invalid_tile",
      };
    }
    for (u32 index = tile.begin; index < tile.end; ++index) {
      visits[index].fetch_add(1u, std::memory_order_relaxed);
    }
    return ComputeTileCallbackResult{};
  });
  if (!run.ok) {
    std::fprintf(stderr, "compute tile boundary run failed: %s\n", run.reason);
  }
  TEST_ASSERT(run.ok);
  TEST_ASSERT(!invalid_tile.load(std::memory_order_relaxed));
  TEST_ASSERT(run.completed_tile_count == expected_tiles);
  TEST_ASSERT(run.worker_tile_count == expected_tiles);
  TEST_ASSERT(run.backend_dispatch_count == 1u);
  TEST_ASSERT(run.worker_count == kWorkers);
  TEST_ASSERT(run.participating_workers >= 2u);
  for (u32 index = 0u; index < count; ++index) {
    TEST_ASSERT(visits[index].load(std::memory_order_relaxed) == 1u);
  }
  TEST_ASSERT(run.last_tile_units == expected_tail);
  return 0;
}

int test_boundaries_and_uneven_tail_are_exact() {
  if (CheckBoundary(kBoundary - 1u, 16u, kTileUnits - 1u) != 0) {
    return 1;
  }
  if (CheckBoundary(kBoundary, 16u, kTileUnits) != 0) {
    return 1;
  }
  return CheckBoundary(kBoundary + 1u, 17u, 1u);
}

int CheckThresholdFreeDispatch(const u32 count) {
  auto pool = kernel_contract_test::BuildStaticPool(kWorkers);
  ComputeTileExecutor executor =
      MakeExecutor(kernel_contract_test::MakeFakeBackend(&pool), kWorkers);
  const auto prepared = executor.prepare(count);
  TEST_ASSERT(prepared.ok);
  TEST_ASSERT(prepared.worker_count == kWorkers);
  TEST_ASSERT(prepared.tile_count != 0u);
  ComputeTileExecutor run_executor = executor.make_run();
  TEST_ASSERT(run_executor.prepared());

  std::array<std::atomic<u32>, kBoundary + 2u> visits{};
  std::atomic<u32> callbacks{0u};
  const auto run = run_executor.run([&](const ComputeTile &tile) {
    callbacks.fetch_add(1u, std::memory_order_relaxed);
    for (u32 index = tile.begin; index < tile.end; ++index) {
      visits[index].fetch_add(1u, std::memory_order_relaxed);
    }
    return ComputeTileCallbackResult{};
  });
  TEST_ASSERT(run.ok);
  TEST_ASSERT(run.worker_count == kWorkers);
  TEST_ASSERT(run.backend_dispatch_count == 1u);
  TEST_ASSERT(run.worker_tile_count == prepared.tile_count);
  TEST_ASSERT(run.completed_tile_count == prepared.tile_count);
  TEST_ASSERT(callbacks.load(std::memory_order_relaxed) == prepared.tile_count);
  TEST_ASSERT(run.participating_workers ==
              std::min(kWorkers, prepared.tile_count));
  for (u32 index = 0u; index < count; ++index) {
    TEST_ASSERT(visits[index].load(std::memory_order_relaxed) == 1u);
  }
  return 0;
}

int test_positive_counts_always_dispatch_through_worker_backend() {
  constexpr std::array counts{
      1u, kTileUnits - 1u, kTileUnits, kTileUnits + 1u, kBoundary + 1u,
  };
  for (const u32 count : counts) {
    if (CheckThresholdFreeDispatch(count) != 0) {
      return 1;
    }
  }
  return 0;
}

struct InvalidProviderContext {
  std::atomic<u32> acquires{0u};
};

struct CountedBackend {
  rund::kernel::WorkerBackend base{};
  std::atomic<u32> dispatches{0u};

  [[nodiscard]] rund::kernel::WorkerBackend backend() noexcept;
};

u32 CountedWorkers(void *raw) {
  auto *const counted = static_cast<CountedBackend *>(raw);
  return counted->base.worker_count(counted->base.context);
}

rund::kernel::WorkerAffinityPolicy CountedAffinity(void *raw) {
  auto *const counted = static_cast<CountedBackend *>(raw);
  return counted->base.affinity_policy(counted->base.context);
}

rund::kernel::WorkerBackendCapabilities CountedCaps(void *raw) {
  auto *const counted = static_cast<CountedBackend *>(raw);
  return counted->base.capabilities != nullptr
             ? counted->base.capabilities(counted->base.context)
             : rund::kernel::WorkerBackendCapabilities{};
}

bool CountedNested(void *raw) {
  auto *const counted = static_cast<CountedBackend *>(raw);
  return counted->base.is_nested != nullptr &&
         counted->base.is_nested(counted->base.context);
}

bool CountedExecute(void *raw, const rund::kernel::Partition *partitions,
                    const u32 partition_count,
                    const rund::kernel::WorkerTask task,
                    rund::kernel::WorkerStats *const stats) {
  auto *const counted = static_cast<CountedBackend *>(raw);
  counted->dispatches.fetch_add(1u, std::memory_order_relaxed);
  return counted->base.execute_partitions(counted->base.context, partitions,
                                          partition_count, task, stats);
}

rund::kernel::WorkerBackend CountedBackend::backend() noexcept {
  return rund::kernel::WorkerBackend{
      .context = this,
      .worker_count = CountedWorkers,
      .affinity_policy = CountedAffinity,
      .capabilities = CountedCaps,
      .is_nested = CountedNested,
      .execute_partitions = CountedExecute,
  };
}

rund::kernel::ParallelRuntime AcquireInvalidProvider(void *raw,
                                                     const u32 workers) {
  auto *const context = static_cast<InvalidProviderContext *>(raw);
  context->acquires.fetch_add(1u, std::memory_order_relaxed);
  return rund::kernel::ParallelRuntime{
      .workers = workers,
      .reason = "test_provider_invalid",
  };
}

int test_host_backend_is_explicit_and_ambient_provider_is_ignored() {
  auto default_pool = kernel_contract_test::BuildStaticPool(kWorkers);
  auto node_pool = kernel_contract_test::BuildStaticPool(kWorkers);
  CountedBackend default_backend{
      .base = kernel_contract_test::MakeFakeBackend(&default_pool)};
  CountedBackend node{.base =
                          kernel_contract_test::MakeFakeBackend(&node_pool)};
  ComputeTileExecutor executor{
      default_backend.backend(), kWorkers,
      rund::kernel::physical_tiles(kTargetTilesPerWorker, 1u, kTileUnits)};
  TEST_ASSERT(executor.prepare(kBoundary).ok);
  InvalidProviderContext context{};
  const rund::kernel::executor_detail::ScopedParallelRuntimeProvider provider{
      rund::kernel::ParallelRuntimeProvider{
          .context = &context,
          .acquire = AcquireInvalidProvider,
      }};
  TEST_ASSERT(provider);
  TEST_ASSERT(rund::kernel::executor_detail::ParallelRuntimeProviderActive());
  ComputeTileExecutor run_executor = executor.make_run();
  TEST_ASSERT(run_executor.prepared());
  std::atomic<u32> callbacks{0u};
  const auto standalone = run_executor.run([&](const ComputeTile &) {
    callbacks.fetch_add(1u, std::memory_order_relaxed);
    return ComputeTileCallbackResult{};
  });
  TEST_ASSERT(standalone.ok);
  TEST_ASSERT(context.acquires.load(std::memory_order_relaxed) == 0u);
  TEST_ASSERT(default_backend.dispatches.load(std::memory_order_relaxed) == 1u);
  TEST_ASSERT(node.dispatches.load(std::memory_order_relaxed) == 0u);

  const u32 standalone_callbacks = callbacks.load(std::memory_order_relaxed);
  const auto hosted =
      run_executor.run_with(node.backend(), [&](const ComputeTile &) {
        callbacks.fetch_add(1u, std::memory_order_relaxed);
        return ComputeTileCallbackResult{};
      });
  TEST_ASSERT(hosted.ok);
  TEST_ASSERT(context.acquires.load(std::memory_order_relaxed) == 0u);
  TEST_ASSERT(default_backend.dispatches.load(std::memory_order_relaxed) == 1u);
  TEST_ASSERT(node.dispatches.load(std::memory_order_relaxed) == 1u);
  TEST_ASSERT(callbacks.load(std::memory_order_relaxed) ==
              standalone_callbacks + hosted.completed_tile_count);

  const auto invalid = run_executor.run_with(
      rund::kernel::WorkerBackend{},
      [](const ComputeTile &) { return ComputeTileCallbackResult{}; });
  TEST_ASSERT(!invalid.ok);
  TEST_ASSERT(std::string_view{invalid.reason} ==
              "compute_tile_backend_invalid");
  return 0;
}

int test_const_callback_lvalue_is_supported() {
  auto pool = kernel_contract_test::BuildStaticPool(1u);
  ComputeTileExecutor executor{kernel_contract_test::MakeFakeBackend(&pool),
                               1u};
  TEST_ASSERT(executor.prepare(4u).ok);
  ComputeTileExecutor run = executor.make_run();
  TEST_ASSERT(run.prepared());
  const auto callback = [](const ComputeTile &) {
    return ComputeTileCallbackResult{};
  };
  TEST_ASSERT(run.run(callback).ok);
  return 0;
}

int test_prepared_plan_makes_independent_run_state() {
  auto pool = kernel_contract_test::BuildStaticPool(kWorkers);
  ComputeTileExecutor plan =
      MakeExecutor(kernel_contract_test::MakeFakeBackend(&pool), kWorkers);
  TEST_ASSERT(plan.prepare(kBoundary + 1u).ok);
  ComputeTileExecutor first = plan.make_run();
  ComputeTileExecutor second = plan.make_run();
  TEST_ASSERT(first.prepared());
  TEST_ASSERT(second.prepared());
  std::array<u32, kBoundary + 1u> first_values{};
  std::array<u32, kBoundary + 1u> second_values{};
  TEST_ASSERT(first
                  .run([&](const ComputeTile &tile) {
                    for (u32 index = tile.begin; index < tile.end; ++index) {
                      first_values[index] = index + 1u;
                    }
                    return tile.index == 1u
                               ? ComputeTileCallbackResult{false,
                                                           "first_run_failed"}
                               : ComputeTileCallbackResult{};
                  })
                  .reason == std::string_view{"first_run_failed"});
  const auto second_result = second.run([&](const ComputeTile &tile) {
    for (u32 index = tile.begin; index < tile.end; ++index) {
      second_values[index] = index + 2u;
    }
    return ComputeTileCallbackResult{};
  });
  TEST_ASSERT(second_result.ok);
  TEST_ASSERT(second_result.completed_tile_count == plan.tile_count());
  for (u32 index = 0u; index < kBoundary + 1u; ++index) {
    TEST_ASSERT(second_values[index] == index + 2u);
  }
  return 0;
}

int test_borrowed_storage_rebinds_without_allocation_or_duplicate_owner() {
  auto pool = kernel_contract_test::BuildStaticPool(kWorkers);
  ComputeTileExecutor source =
      MakeExecutor(kernel_contract_test::MakeFakeBackend(&pool), kWorkers);
  TEST_ASSERT(source.prepare(kBoundary + 1u).ok);
  const auto first_plan = source.run_plan();
  TEST_ASSERT(first_plan.prepared());
  TEST_ASSERT(first_plan.count() == kBoundary + 1u);
  TEST_ASSERT(first_plan.tile_count() == 17u);

  const auto direct = source.run(
      [](const ComputeTile &) { return ComputeTileCallbackResult{}; });
  TEST_ASSERT(!direct.ok);
  TEST_ASSERT(std::string_view{direct.reason} ==
              "compute_tile_run_storage_missing");

  ExternalRunStorage<17u, kWorkers> external{};
  kernel_contract_test::memory_allocation::Reset();
  ComputeTileExecutor first = first_plan.bind(external.view());
  kernel_contract_test::memory_allocation::Stop();
  TEST_ASSERT(kernel_contract_test::memory_allocation::Count() == 0u);
  TEST_ASSERT(first.prepared());
  TEST_ASSERT(first.has_run_storage());
  TEST_ASSERT(first.borrowed_run_storage());
  TEST_ASSERT(!external.state.busy());
  const u64 first_generation = external.state.generation();
  TEST_ASSERT(first_generation != 0u);

  kernel_contract_test::memory_allocation::Reset();
  const auto first_result = first.run(
      [](const ComputeTile &) { return ComputeTileCallbackResult{}; });
  kernel_contract_test::memory_allocation::Stop();
  TEST_ASSERT(first_result.ok);
  TEST_ASSERT(kernel_contract_test::memory_allocation::Count() == 0u);

  TEST_ASSERT(source.prepare(kTileUnits + 1u).ok);
  const auto second_plan = source.run_plan();
  TEST_ASSERT(first_plan.count() == kBoundary + 1u);
  TEST_ASSERT(second_plan.count() == kTileUnits + 1u);
  const auto envelope = rund::kernel::MergeComputeTileRunStoragePlans(
      first_plan.storage_plan(), second_plan.storage_plan());
  TEST_ASSERT(envelope.ok);
  TEST_ASSERT(envelope.failure_slot_capacity == 17u);
  TEST_ASSERT(envelope.worker_capacity == kWorkers);

  ComputeTileExecutor second = second_plan.bind(external.view());
  TEST_ASSERT(second.prepared());
  TEST_ASSERT(external.state.generation() == first_generation + 1u);
  TEST_ASSERT(!first.prepared());
  const auto rebound = first.run(
      [](const ComputeTile &) { return ComputeTileCallbackResult{}; });
  TEST_ASSERT(!rebound.ok);
  TEST_ASSERT(std::string_view{rebound.reason} == "compute_tile_run_rebound");
  TEST_ASSERT(
      second
          .run([](const ComputeTile &) { return ComputeTileCallbackResult{}; })
          .ok);

  ExternalRunStorage<16u, kWorkers> undersized{};
  ComputeTileExecutor rejected = first_plan.bind(undersized.view());
  TEST_ASSERT(!rejected.prepared());
  const auto capacity = rejected.run(
      [](const ComputeTile &) { return ComputeTileCallbackResult{}; });
  TEST_ASSERT(std::string_view{capacity.reason} ==
              "compute_tile_run_storage_capacity");
  return 0;
}

int test_bounded_bind_clamps_frozen_max_plan_without_allocation() {
  constexpr u32 active_count = kBoundary - 1u;
  constexpr u32 active_tiles = 16u;
  constexpr u32 active_tail = kTileUnits - 1u;
  auto pool = kernel_contract_test::BuildStaticPool(kWorkers);
  ComputeTileExecutor source =
      MakeExecutor(kernel_contract_test::MakeFakeBackend(&pool), kWorkers);
  TEST_ASSERT(source.prepare(kBoundary + 1u).ok);
  const auto plan = source.run_plan();
  ExternalRunStorage<17u, kWorkers> external{};

  kernel_contract_test::memory_allocation::Reset();
  ComputeTileExecutor bounded = plan.bind(external.view(), active_count);
  kernel_contract_test::memory_allocation::Stop();
  TEST_ASSERT(kernel_contract_test::memory_allocation::Count() == 0u);
  TEST_ASSERT(bounded.prepared());
  TEST_ASSERT(bounded.count() == active_count);
  TEST_ASSERT(bounded.tile_count() == active_tiles);
  TEST_ASSERT(plan.count() == kBoundary + 1u);
  TEST_ASSERT(plan.tile_count() == 17u);

  std::array<std::atomic<u32>, kBoundary + 2u> visits{};
  std::atomic<bool> invalid_tile{false};
  kernel_contract_test::memory_allocation::Reset();
  const auto result = bounded.run([&](const ComputeTile &tile) {
    if (tile.begin >= tile.end || tile.end > active_count) {
      invalid_tile.store(true, std::memory_order_relaxed);
      return ComputeTileCallbackResult{.ok = false,
                                       .reason = "test_invalid_tile"};
    }
    for (u32 index = tile.begin; index < tile.end; ++index) {
      visits[index].fetch_add(1u, std::memory_order_relaxed);
    }
    return ComputeTileCallbackResult{};
  });
  kernel_contract_test::memory_allocation::Stop();
  TEST_ASSERT(result.ok);
  TEST_ASSERT(!invalid_tile.load(std::memory_order_relaxed));
  TEST_ASSERT(result.completed_tile_count == active_tiles);
  TEST_ASSERT(result.worker_tile_count == active_tiles);
  TEST_ASSERT(result.last_tile_units == active_tail);
  TEST_ASSERT(kernel_contract_test::memory_allocation::Count() == 0u);
  for (u32 index = 0u; index < active_count; ++index) {
    TEST_ASSERT(visits[index].load(std::memory_order_relaxed) == 1u);
  }
  for (u32 index = active_count; index < visits.size(); ++index) {
    TEST_ASSERT(visits[index].load(std::memory_order_relaxed) == 0u);
  }

  ComputeTileExecutor empty = plan.bind(external.view(), 0u);
  TEST_ASSERT(empty.prepared());
  TEST_ASSERT(empty.count() == 0u);
  TEST_ASSERT(empty.tile_count() == 0u);
  TEST_ASSERT(empty
                  .run([](const ComputeTile &) {
                    return ComputeTileCallbackResult{.ok = false,
                                                     .reason = "must_not_run"};
                  })
                  .ok);

  ComputeTileExecutor invalid = plan.bind(external.view(), kBoundary + 2u);
  TEST_ASSERT(!invalid.prepared());
  const auto rejected = invalid.run(
      [](const ComputeTile &) { return ComputeTileCallbackResult{}; });
  TEST_ASSERT(std::string_view{rejected.reason} ==
              "compute_tile_run_active_count_invalid");

  ComputeTileExecutor untiled_source{
      kernel_contract_test::MakeFakeBackend(&pool), kWorkers,
      rund::kernel::no_physical_tiles()};
  TEST_ASSERT(untiled_source.prepare(kBoundary + 1u).ok);
  const auto untiled_plan = untiled_source.run_plan();
  TEST_ASSERT(untiled_plan.tile_count() == kWorkers);
  ExternalRunStorage<kWorkers, kWorkers> untiled_external{};
  ComputeTileExecutor untiled = untiled_plan.bind(untiled_external.view(), 1u);
  TEST_ASSERT(untiled.prepared());
  TEST_ASSERT(untiled.count() == 1u);
  TEST_ASSERT(untiled.tile_count() == 1u);
  u32 untiled_visits = 0u;
  const auto untiled_result = untiled.run([&](const ComputeTile &tile) {
    if (tile.index != 0u || tile.begin != 0u || tile.end != 1u) {
      return ComputeTileCallbackResult{.ok = false,
                                       .reason = "test_invalid_tile"};
    }
    ++untiled_visits;
    return ComputeTileCallbackResult{};
  });
  TEST_ASSERT(untiled_result.ok);
  TEST_ASSERT(untiled_result.completed_tile_count == 1u);
  TEST_ASSERT(untiled_result.last_tile_units == 1u);
  TEST_ASSERT(untiled_visits == 1u);
  return 0;
}

int test_retained_memory_is_exact_stable_and_allocation_free() {
  ComputeTileExecutor empty{};
  const ComputeTileRetainedMemory empty_memory = empty.retained_memory();
  const auto empty_run_memory = empty.planned_run_memory();
  TEST_ASSERT(empty_memory.state_bytes == 0u);
  TEST_ASSERT(empty_memory.workspace_bytes == 0u);
  TEST_ASSERT(empty_memory.failure_slot_bytes == 0u);
  TEST_ASSERT(empty_memory.worker_tile_bytes == 0u);
  TEST_ASSERT(empty_memory.async_context_bytes == 0u);
  TEST_ASSERT(empty_memory.total_bytes == 0u);
  TEST_ASSERT(!empty_run_memory.ok);
  TEST_ASSERT(std::string_view{empty_run_memory.reason} ==
              "compute_tile_run_memory_not_prepared");
  TEST_ASSERT(empty_run_memory.memory.total_bytes == 0u);

  constexpr auto overflow = PlanComputeTileRunMemory(ComputeTileRetainedMemory{
      .state_bytes = std::numeric_limits<u64>::max(),
      .workspace_bytes = 1u,
  });
  static_assert(!overflow.ok);
  static_assert(overflow.memory.total_bytes == std::numeric_limits<u64>::max());
  TEST_ASSERT(std::string_view{overflow.reason} ==
              "compute_tile_run_memory_overflow");

  auto pool = kernel_contract_test::BuildStaticPool(kWorkers);
  ComputeTileExecutor plan =
      MakeExecutor(kernel_contract_test::MakeFakeBackend(&pool), kWorkers);
  const ComputeTileRetainedMemory constructed = plan.retained_memory();
  const auto unprepared_run_memory = plan.planned_run_memory();
  TEST_ASSERT(constructed.state_bytes == 0u);
  TEST_ASSERT(constructed.total_bytes == TotalBytes(constructed));
  TEST_ASSERT(!unprepared_run_memory.ok);
  TEST_ASSERT(std::string_view{unprepared_run_memory.reason} ==
              "compute_tile_run_memory_not_prepared");
  const auto prepared = plan.prepare(kBoundary + 1u);
  TEST_ASSERT(prepared.ok);
  const ComputeTileRetainedMemory planned = plan.retained_memory();
  TEST_ASSERT(planned.state_bytes != 0u);
  TEST_ASSERT(planned.workspace_bytes != 0u);
  TEST_ASSERT(planned.failure_slot_bytes == 0u);
  TEST_ASSERT(planned.worker_tile_bytes == 0u);
  TEST_ASSERT(planned.async_context_bytes == 0u);
  TEST_ASSERT(planned.total_bytes == TotalBytes(planned));

  kernel_contract_test::memory_allocation::Reset();
  const auto first_run_plan = plan.planned_run_memory();
  const auto second_run_plan = plan.planned_run_memory();
  kernel_contract_test::memory_allocation::Stop();
  TEST_ASSERT(kernel_contract_test::memory_allocation::Count() == 0u);
  TEST_ASSERT(first_run_plan.ok);
  TEST_ASSERT(second_run_plan.ok);
  TEST_ASSERT(std::string_view{first_run_plan.reason} == "pass");
  TEST_ASSERT(SameMemory(first_run_plan.memory, second_run_plan.memory));
  TEST_ASSERT(first_run_plan.memory.state_bytes != 0u);
  TEST_ASSERT(first_run_plan.memory.workspace_bytes ==
              static_cast<u64>(prepared.worker_count) *
                  (sizeof(u32) + 3u * sizeof(u64)));
  TEST_ASSERT(first_run_plan.memory.failure_slot_bytes ==
              static_cast<u64>(prepared.tile_count) * sizeof(const char *));
  TEST_ASSERT(first_run_plan.memory.worker_tile_bytes ==
              static_cast<u64>(prepared.worker_count) * sizeof(u32));
  TEST_ASSERT(first_run_plan.memory.async_context_bytes == 0u);

  kernel_contract_test::memory_allocation::Reset();
  ComputeTileExecutor run = plan.make_run();
  kernel_contract_test::memory_allocation::Stop();
  TEST_ASSERT(run.prepared());
  TEST_ASSERT(kernel_contract_test::memory_allocation::Count() == 7u);
  const ComputeTileRetainedMemory retained = run.retained_memory();
  TEST_ASSERT(SameMemory(retained, first_run_plan.memory));
  TEST_ASSERT(retained.total_bytes == TotalBytes(retained));

  kernel_contract_test::memory_allocation::Reset();
  const ComputeTileRetainedMemory first = plan.retained_memory();
  const ComputeTileRetainedMemory second = run.retained_memory();
  kernel_contract_test::memory_allocation::Stop();
  TEST_ASSERT(kernel_contract_test::memory_allocation::Count() == 0u);
  TEST_ASSERT(SameMemory(first, planned));
  TEST_ASSERT(SameMemory(second, retained));

  kernel_contract_test::memory_allocation::Reset();
  const auto hot =
      run.run([](const ComputeTile &) { return ComputeTileCallbackResult{}; });
  kernel_contract_test::memory_allocation::Stop();
  TEST_ASSERT(hot.ok);
  TEST_ASSERT(kernel_contract_test::memory_allocation::Count() == 0u);
  TEST_ASSERT(SameMemory(run.retained_memory(), retained));

  ComputeTileExecutor moved = std::move(run);
  TEST_ASSERT(SameMemory(moved.retained_memory(), retained));
  TEST_ASSERT(run.retained_memory().total_bytes == 0u);
  return 0;
}

} // namespace

int RunComputeTileContract() {
  if (test_positive_counts_always_dispatch_through_worker_backend() != 0) {
    return 1;
  }
  if (test_boundaries_and_uneven_tail_are_exact() != 0) {
    return 1;
  }
  if (test_const_callback_lvalue_is_supported() != 0) {
    return 1;
  }
  if (test_prepared_plan_makes_independent_run_state() != 0) {
    return 1;
  }
  if (test_borrowed_storage_rebinds_without_allocation_or_duplicate_owner() !=
      0) {
    return 1;
  }
  if (test_bounded_bind_clamps_frozen_max_plan_without_allocation() != 0) {
    return 1;
  }
  if (test_retained_memory_is_exact_stable_and_allocation_free() != 0) {
    return 1;
  }
  return test_host_backend_is_explicit_and_ambient_provider_is_ignored();
}

} // namespace program_compute_contract
