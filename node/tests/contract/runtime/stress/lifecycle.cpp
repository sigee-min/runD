#include "test/assert.hpp"

#include <kernel/dispatch/orchestrator.hpp>
#include <kernel/program/executor.hpp>
#include <kernel/program/executor/policy.hpp>
#include <kernel/program/skeleton.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

rund::task::Task<void> YieldOnce(std::atomic<std::uint32_t> *const count) {
  (void)co_await rund::task::yield();
  count->fetch_add(1u, std::memory_order_relaxed);
}

std::atomic<std::uint32_t> function_calls{0u};

void ScopeFunction() {
  function_calls.fetch_add(1u, std::memory_order_relaxed);
}

struct ScopeCallbackProbe final {
  std::uint32_t *calls = nullptr;
  std::uint32_t *copies = nullptr;
  std::uint32_t *moves = nullptr;

  ScopeCallbackProbe(std::uint32_t &call_count, std::uint32_t &copy_count,
                     std::uint32_t &move_count) noexcept
      : calls(&call_count), copies(&copy_count), moves(&move_count) {}

  ScopeCallbackProbe(const ScopeCallbackProbe &other) noexcept
      : calls(other.calls), copies(other.copies), moves(other.moves) {
    ++*copies;
  }

  ScopeCallbackProbe(ScopeCallbackProbe &&other) noexcept
      : calls(other.calls), copies(other.copies), moves(other.moves) {
    ++*moves;
  }

  void operator()() const noexcept { ++*calls; }
};

rund::SessionConfig ScopeOptions(const std::uint64_t id) {
  rund::SessionConfig options{};
  options.id = id;
  options.workers = 1u;
  options.scheduler.task_workers = 1u;
  options.scheduler.task_capacity = 8u;
  options.scheduler.ready_queue_capacity = 8u;
  return options;
}

} // namespace

int CheckRuntimeLifecycleStress() {
  constexpr std::uint32_t capacity = 128u;
  const unsigned hardware = std::thread::hardware_concurrency();
  const std::uint32_t workers = hardware == 0u ? 1u : hardware;
  rund::SessionConfig saturated_options{};
  saturated_options.id = 900u;
  saturated_options.workers = workers;
  saturated_options.scheduler.task_workers = workers;
  saturated_options.scheduler.task_capacity = capacity;
  saturated_options.scheduler.ready_queue_capacity = capacity;
  rund::Session saturated{};
  TEST_ASSERT(saturated.open(saturated_options));
  std::atomic<std::uint32_t> saturated_completed{0u};
  bool saturated_joined = true;
  const auto saturated_scope = saturated.scope([&] {
    std::vector<rund::task::Handle> tasks{};
    tasks.reserve(capacity - 1u);
    for (std::uint32_t index = 0u; index + 1u < capacity; ++index) {
      auto task =
          rund::task::spawn("capacity-near", YieldOnce(&saturated_completed));
      saturated_joined = saturated_joined && static_cast<bool>(task);
      tasks.push_back(std::move(task));
    }
    for (const auto &task : tasks) {
      saturated_joined =
          saturated_joined && static_cast<bool>(rund::task::join(task));
    }
  });
  TEST_ASSERT(saturated_scope);
  TEST_ASSERT(saturated_joined);
  TEST_ASSERT(saturated_completed.load(std::memory_order_relaxed) ==
              capacity - 1u);
  TEST_ASSERT(
      saturated_scope.tasks().resources().coroutine_frames_high_water() ==
      capacity - 1u);
  TEST_ASSERT(saturated_scope.tasks().participating_task_workers() == workers);
  TEST_ASSERT(saturated.close());

  std::atomic<std::uint32_t> completed{0u};
  for (std::uint32_t iteration = 0u; iteration < 64u; ++iteration) {
    rund::SessionConfig options{};
    options.id = iteration + 1u;
    options.workers = 1u;
    options.scheduler.task_workers = 1u;
    options.scheduler.task_capacity = 4u;
    options.scheduler.ready_queue_capacity = 4u;
    rund::Session runtime{};
    TEST_ASSERT(runtime.open(options));
    bool joined = false;
    const auto scope = runtime.scope([&] {
      const auto task = rund::task::spawn("lifecycle", YieldOnce(&completed));
      joined = static_cast<bool>(rund::task::join(task));
    });
    TEST_ASSERT(scope);
    TEST_ASSERT(joined);
    TEST_ASSERT(runtime.close());
  }
  TEST_ASSERT(completed.load(std::memory_order_relaxed) == 64u);

  rund::Session callback_runtime{};
  TEST_ASSERT(callback_runtime.open(ScopeOptions(1001u)));

  std::uint32_t callback_calls = 0u;
  std::uint32_t callback_copies = 0u;
  std::uint32_t callback_moves = 0u;
  ScopeCallbackProbe callback{callback_calls, callback_copies, callback_moves};
  TEST_ASSERT(callback_runtime.scope(callback));
  TEST_ASSERT(callback_runtime.scope(std::move(callback)));
  function_calls.store(0u, std::memory_order_relaxed);
  TEST_ASSERT(callback_runtime.scope(ScopeFunction));
  TEST_ASSERT(callback_calls == 2u);
  TEST_ASSERT(callback_copies == 0u);
  TEST_ASSERT(callback_moves == 0u);
  TEST_ASSERT(function_calls.load(std::memory_order_relaxed) == 1u);

  rund::Session::Result busy_scope{};
  const rund::Session::Result outer_scope = callback_runtime.scope(
      [&] { busy_scope = callback_runtime.scope([] {}); });
  TEST_ASSERT(outer_scope);
  TEST_ASSERT(!busy_scope);
  TEST_ASSERT(busy_scope.error() ==
              std::string_view{"runtime_reentry_forbidden"});

  rund::Session nested_runtime{};
  TEST_ASSERT(nested_runtime.open(ScopeOptions(1002u)));
  rund::Session::Result nested_scope{};
  rund::task::Status restored_join{};
  rund::kernel::SkeletonResult restored_parallel{};
  std::array<rund::kernel::i32, 1u> restored_value{};
  const rund::Session::Result restored_scope = callback_runtime.scope([&] {
    nested_scope = nested_runtime.scope([] {});
    restored_join =
        rund::task::join(rund::task::spawn("restored-scheduler", [] {}));
    auto view = rund::kernel::view<rund::kernel::i32>(
        restored_value.data(), rund::kernel::Index<1u>{restored_value.size()});
    restored_parallel = rund::kernel::each(
        rund::kernel::par(), rund::kernel::space(restored_value.size()),
        [&](auto index) noexcept { view(index) = 7; });
  });
  TEST_ASSERT(restored_scope);
  TEST_ASSERT(nested_scope);
  TEST_ASSERT(restored_join);
  TEST_ASSERT(restored_parallel.ok);
  TEST_ASSERT(restored_value[0] == 7);
  TEST_ASSERT(!rund::task::spawn("scope-restored", [] {}));

  const rund::Session::Result failed_scope = callback_runtime.scope([] {
    TEST_ASSERT(rund::task::spawn("scope-drain-failure", [] { throw 1; }));
    throw 2;
  });
  TEST_ASSERT(!failed_scope);
  TEST_ASSERT(failed_scope.error() ==
              std::string_view{"runtime_scope_callback_failed"});
  const rund::task::Handle missing_after_failure =
      rund::task::spawn("scope-failure-restored", [] {});
  TEST_ASSERT(!missing_after_failure);
  TEST_ASSERT(missing_after_failure.error() ==
              std::string_view{"node_runtime_missing"});
  const rund::kernel::SkeletonResult provider_after_failure =
      rund::kernel::each(rund::kernel::par(), rund::kernel::space(1u),
                         [](auto) noexcept {});
  TEST_ASSERT(!provider_after_failure.ok);
  TEST_ASSERT(provider_after_failure.reason ==
              std::string_view{"parallel_runtime_missing"});

  TEST_ASSERT(nested_runtime.close());
  TEST_ASSERT(callback_runtime.close());
  return 0;
}
