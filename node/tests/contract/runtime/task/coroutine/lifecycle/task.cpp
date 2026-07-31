#include <rund/task/api.hpp>
#include <rund/task/await.hpp>
#include "../local.hpp"

namespace rund::node::test_contract::coroutine {

rund::task::Task<void> SleepChild() {
  (void)co_await rund::task::sleep(std::chrono::nanoseconds{1});
}

rund::task::Task<void> YieldChild() { (void)co_await rund::task::yield(); }

rund::task::Task<void>
JoinAwait(const rund::task::Handle child,
          std::atomic<std::uint32_t>* const after_await) {
  const rund::task::Status joined = co_await child;
  if (joined) {
    after_await->fetch_add(1u, std::memory_order_release);
  }
}

rund::task::Task<void>
HandleAwait(std::atomic<std::uint32_t>* const after_await) {
  const rund::task::Status child = co_await rund::task::spawn(
      "coroutine-handle-await-child", YieldChild());
  if (child) {
    after_await->fetch_add(1u, std::memory_order_release);
  }
}

rund::task::Task<void> NestedComplete() {
  const rund::task::Status yielded = co_await rund::task::yield();
  (void)yielded;
  const rund::task::Status slept =
      co_await rund::task::sleep(std::chrono::nanoseconds{1});
  (void)slept;
  co_return;
}

rund::task::Task<void>
NestedTaskAwait(std::atomic<std::uint32_t>* const after_await) {
  const rund::task::Result<void> nested = co_await NestedComplete();
  if (nested) {
    after_await->fetch_add(1u, std::memory_order_release);
  }
}

rund::task::Task<int> SuspendsWithValue() {
  const rund::task::Status yielded = co_await rund::task::yield();
  if (!yielded) {
    co_return -1;
  }
  co_return 7;
}

rund::task::Task<void> AwaitValue(std::atomic<std::uint32_t>* const value) {
  const rund::task::Result<int> result = co_await SuspendsWithValue();
  if (result) {
    value->store(static_cast<std::uint32_t>(*result),
                 std::memory_order_release);
  }
}

rund::task::Task<void> AwaitMany(std::atomic<std::uint32_t>* const total) {
  for (std::uint32_t index = 0u; index < 32u; ++index) {
    const rund::task::Result<int> result = co_await SuspendsWithValue();
    if (!result) {
      co_return;
    }
    total->fetch_add(static_cast<std::uint32_t>(*result),
                     std::memory_order_release);
  }
}

int CheckCoroutineJoinAwait() {
  std::atomic<std::uint32_t> after{0u};
  rund::task::Status joined{};
  bool handle_valid = false;
  const rund::Session::Result report =
      rund::run(rund::SessionConfig{
        .id = 792u,
        .workers = 2u,
        .scheduler = {
          .task_capacity = 4u,
          .ready_queue_capacity = 4u,
        },
      },
                [&] {
                  const rund::task::Handle child = rund::task::spawn(
                      "coroutine-join-await-child", SleepChild());
                  const rund::task::Handle task = rund::task::spawn(
                      "coroutine-join", JoinAwait(child, &after));
                  handle_valid = static_cast<bool>(task);
                  joined = rund::task::join(task);
                });
  return AssertCoroutineAwaitSuccess(report, joined, handle_valid, after, 2u,
                                     2u);
}

int CheckCoroutineHandleAwait() {
  std::atomic<std::uint32_t> after{0u};
  rund::task::Status joined{};
  bool handle_valid = false;
  const rund::Session::Result report =
      rund::run(rund::SessionConfig{
        .id = 793u,
        .workers = 2u,
        .scheduler = {
          .task_capacity = 4u,
          .ready_queue_capacity = 4u,
        },
      },
                [&] {
                  const rund::task::Handle task = rund::task::spawn(
                      "coroutine-handle", HandleAwait(&after));
                  handle_valid = static_cast<bool>(task);
                  joined = rund::task::join(task);
                });
  return AssertCoroutineAwaitSuccess(report, joined, handle_valid, after, 2u,
                                     2u);
}

int CheckCoroutineNestedTask() {
  std::atomic<std::uint32_t> after{0u};
  rund::task::Status joined{};
  bool handle_valid = false;
  const rund::Session::Result report =
      rund::run(rund::SessionConfig{
        .id = 794u,
        .workers = 2u,
        .scheduler = {
          .task_capacity = 4u,
          .ready_queue_capacity = 4u,
        },
      },
                [&] {
                  const rund::task::Handle task =
                      rund::task::spawn("coroutine-nested-task",
                                        NestedTaskAwait(&after));
                  handle_valid = static_cast<bool>(task);
                  joined = rund::task::join(task);
                });
  return AssertCoroutineAwaitSuccess(report, joined, handle_valid, after, 2u,
                                     2u);
}

int CheckNestedResultReuseAndLeafBoundary() {
  std::atomic<std::uint32_t> nested_value{0u};
  rund::task::Status nested_join{};
  const rund::Session::Result nested = rund::run(
      rund::SessionConfig{
        .id = 811u,
        .workers = 2u,
        .scheduler = {
          .task_capacity = 8u,
          .ready_queue_capacity = 8u,
        },
      },
      [&] {
        const rund::task::Handle task =
            rund::task::spawn("coroutine-nested-value", AwaitValue(&nested_value));
        nested_join = rund::task::join(task);
      });
  TEST_ASSERT(nested.ok());
  TEST_ASSERT(nested_join.ok());
  TEST_ASSERT(nested_value.load(std::memory_order_acquire) == 7u);
  TEST_ASSERT(nested.tasks().coroutine_completions() == 2u);
  TEST_ASSERT(nested.tasks().coroutine_frame_destroys() == 2u);

  std::atomic<std::uint32_t> reused_value{0u};
  rund::task::Status reused_join{};
  const rund::Session::Result reused = rund::run(
      rund::SessionConfig{
        .id = 813u,
        .workers = 1u,
        .scheduler = {
          .task_capacity = 2u,
          .ready_queue_capacity = 2u,
        },
      },
      [&] {
        const rund::task::Handle task =
            rund::task::spawn("coroutine-nested-reuse", AwaitMany(&reused_value));
        reused_join = rund::task::join(task);
      });
  TEST_ASSERT(reused.ok());
  TEST_ASSERT(reused_join.ok());
  TEST_ASSERT(reused_value.load(std::memory_order_acquire) == 224u);
  TEST_ASSERT(reused.tasks().coroutine_completions() == 33u);
  TEST_ASSERT(reused.tasks().coroutine_frame_destroys() == 33u);
  TEST_ASSERT(reused.tasks().task_record_retires() == 33u);

  rund::task::Status leaf_primitive{};
  rund::task::Status leaf_join{};
  const rund::Session::Result leaf = rund::run(
      rund::SessionConfig{
        .id = 812u,
        .workers = 1u,
        .scheduler = {
          .task_capacity = 2u,
          .ready_queue_capacity = 2u,
        },
      },
      [&] {
        const rund::task::Handle task = rund::task::spawn("leaf-join-forbidden", [&] {
          leaf_primitive = rund::task::join(rund::task::Handle{});
        });
        leaf_join = rund::task::join(task);
      });
  TEST_ASSERT(leaf.ok());
  TEST_ASSERT(!leaf_primitive.ok());
  TEST_ASSERT(leaf_primitive.code() ==
              ReasonCode::TaskLeafPrimitiveForbidden);
  TEST_ASSERT(!leaf_join.ok());
  TEST_ASSERT(leaf_join.code() == ReasonCode::TaskLeafPrimitiveForbidden);
  return 0;
}

}  // namespace rund::node::test_contract::coroutine
