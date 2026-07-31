#include "contract/program/compute/local.hpp"
#include "contract/support/allocation.hpp"
#include "contract/support/backend/factory.hpp"
#include "test/assert.hpp"

#include <kernel/program/compute/tile/run.hpp>
#include <kernel/program/executor.hpp>

#include <array>
#include <atomic>
#include <cstdio>
#include <string_view>
#include <utility>

namespace program_compute_contract {
namespace {

using rund::kernel::ComputeTile;
using rund::kernel::ComputeTileCallbackResult;
using rund::kernel::ComputeTileExecutor;
using rund::kernel::ComputeTileRetainedMemory;
using rund::kernel::u32;
using rund::kernel::u64;

constexpr u32 kWorkers = 4u;
constexpr u32 kTileUnits = 32u;
constexpr u32 kTargetTilesPerWorker = 4u;
constexpr u32 kBoundary = kWorkers * kTargetTilesPerWorker * kTileUnits;

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

  std::array<std::atomic<u32>, kBoundary + 2u> visits{};
  std::atomic<bool> invalid_tile{false};
  const auto run = executor.run([&](const ComputeTile &tile) {
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

  std::array<std::atomic<u32>, kBoundary + 2u> visits{};
  std::atomic<u32> callbacks{0u};
  const auto run = executor.run([&](const ComputeTile &tile) {
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
  std::atomic<u32> callbacks{0u};
  const auto standalone = executor.run([&](const ComputeTile &) {
    callbacks.fetch_add(1u, std::memory_order_relaxed);
    return ComputeTileCallbackResult{};
  });
  TEST_ASSERT(standalone.ok);
  TEST_ASSERT(context.acquires.load(std::memory_order_relaxed) == 0u);
  TEST_ASSERT(default_backend.dispatches.load(std::memory_order_relaxed) == 1u);
  TEST_ASSERT(node.dispatches.load(std::memory_order_relaxed) == 0u);

  const u32 standalone_callbacks = callbacks.load(std::memory_order_relaxed);
  const auto hosted =
      executor.run_with(node.backend(), [&](const ComputeTile &) {
        callbacks.fetch_add(1u, std::memory_order_relaxed);
        return ComputeTileCallbackResult{};
      });
  TEST_ASSERT(hosted.ok);
  TEST_ASSERT(context.acquires.load(std::memory_order_relaxed) == 0u);
  TEST_ASSERT(default_backend.dispatches.load(std::memory_order_relaxed) == 1u);
  TEST_ASSERT(node.dispatches.load(std::memory_order_relaxed) == 1u);
  TEST_ASSERT(callbacks.load(std::memory_order_relaxed) ==
              standalone_callbacks + hosted.completed_tile_count);

  const auto invalid =
      executor.run_with(rund::kernel::WorkerBackend{}, [](const ComputeTile &) {
        return ComputeTileCallbackResult{};
      });
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
  const auto callback = [](const ComputeTile &) {
    return ComputeTileCallbackResult{};
  };
  TEST_ASSERT(executor.run(callback).ok);
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

int test_retained_memory_is_exact_stable_and_allocation_free() {
  ComputeTileExecutor empty{};
  const ComputeTileRetainedMemory empty_memory = empty.retained_memory();
  TEST_ASSERT(empty_memory.state_bytes == 0u);
  TEST_ASSERT(empty_memory.workspace_bytes == 0u);
  TEST_ASSERT(empty_memory.failure_slot_bytes == 0u);
  TEST_ASSERT(empty_memory.worker_tile_bytes == 0u);
  TEST_ASSERT(empty_memory.async_context_bytes == 0u);
  TEST_ASSERT(empty_memory.total_bytes == 0u);

  auto pool = kernel_contract_test::BuildStaticPool(kWorkers);
  ComputeTileExecutor plan =
      MakeExecutor(kernel_contract_test::MakeFakeBackend(&pool), kWorkers);
  const ComputeTileRetainedMemory constructed = plan.retained_memory();
  TEST_ASSERT(constructed.state_bytes != 0u);
  TEST_ASSERT(constructed.total_bytes == TotalBytes(constructed));
  const auto prepared = plan.prepare(kBoundary + 1u);
  TEST_ASSERT(prepared.ok);
  const ComputeTileRetainedMemory planned = plan.retained_memory();
  TEST_ASSERT(planned.state_bytes == constructed.state_bytes);
  TEST_ASSERT(planned.workspace_bytes != 0u);
  TEST_ASSERT(planned.failure_slot_bytes >=
              static_cast<u64>(prepared.tile_count) * sizeof(const char *));
  TEST_ASSERT(planned.worker_tile_bytes >=
              static_cast<u64>(prepared.worker_count) * sizeof(u32));
  TEST_ASSERT(planned.async_context_bytes == 0u);
  TEST_ASSERT(planned.total_bytes == TotalBytes(planned));

  ComputeTileExecutor run = plan.make_run();
  TEST_ASSERT(run.prepared());
  const ComputeTileRetainedMemory retained = run.retained_memory();
  TEST_ASSERT(retained.state_bytes == planned.state_bytes);
  TEST_ASSERT(retained.workspace_bytes >= planned.workspace_bytes);
  TEST_ASSERT(retained.failure_slot_bytes >=
              static_cast<u64>(prepared.tile_count) * sizeof(const char *));
  TEST_ASSERT(retained.worker_tile_bytes >=
              static_cast<u64>(prepared.worker_count) * sizeof(u32));
  TEST_ASSERT(retained.async_context_bytes != 0u);
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
  if (test_retained_memory_is_exact_stable_and_allocation_free() != 0) {
    return 1;
  }
  return test_host_backend_is_explicit_and_ambient_provider_is_ignored();
}

} // namespace program_compute_contract
