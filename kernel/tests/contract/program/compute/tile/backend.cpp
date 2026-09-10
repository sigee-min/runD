#include "local.hpp"

#include <atomic>
#include <string_view>

namespace program_compute_contract::tile_contract {
namespace {

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

} // namespace

int CheckExplicitBackend() {
  auto default_pool = kernel_contract_test::BuildStaticPool(Workers);
  auto node_pool = kernel_contract_test::BuildStaticPool(Workers);
  CountedBackend default_backend{
      .base = kernel_contract_test::MakeFakeBackend(&default_pool)};
  CountedBackend node{.base =
                          kernel_contract_test::MakeFakeBackend(&node_pool)};
  ComputeTileExecutor executor{
      default_backend.backend(), Workers,
      rund::kernel::physical_tiles(TargetTilesPerWorker, 1u, TileUnits)};
  TEST_ASSERT(executor.prepare(Boundary).ok);
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

} // namespace program_compute_contract::tile_contract
