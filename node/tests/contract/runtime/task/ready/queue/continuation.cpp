#include "local.hpp"

#include "test/assert.hpp"

#include <rund/session.hpp>
#include <rund/task/api.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>

namespace ready_queue_detail {
namespace {

rund::task::Task<void> YieldRounds(std::uint32_t *const segments,
                                   const std::uint32_t rounds) {
  for (std::uint32_t round = 0u; round < rounds; ++round) {
    ++*segments;
    static_cast<void>(co_await rund::task::yield());
  }
  ++*segments;
}

rund::task::Task<void> CompleteAfterTimer(bool *const completed) {
  const rund::task::Status slept =
      co_await rund::task::sleep(std::chrono::milliseconds{1});
  if (slept) {
    *completed = true;
  }
}

rund::task::Task<void> AwaitAtReadyLimit(
    std::atomic<std::uint64_t> *const queued,
    std::atomic<std::uint64_t> *const continued,
    std::atomic<bool> *const overflow_rejected) {
  const rund::task::Handle pressure =
      rund::task::spawn("ready-pressure", [queued] {
        queued->fetch_add(1u, std::memory_order_relaxed);
      });
  if (!pressure) {
    co_return;
  }
  const rund::task::Handle overflow =
      rund::task::spawn("ready-pressure-overflow", [] {});
  overflow_rejected->store(
      !overflow &&
          overflow.code() ==
              rund::ReasonCode::ReadyQueueCapacityExceeded,
      std::memory_order_relaxed);
  const rund::task::Result<void> child =
      co_await CompleteIndex(continued);
  if (!child) {
    co_return;
  }
}

rund::task::Task<void> AwaitAtTaskLimit(
    std::atomic<std::uint64_t> *const unexpected_child,
    std::atomic<bool> *const child_rejected) {
  const rund::task::Handle pressure =
      rund::task::spawn("task-capacity-pressure", [] {});
  if (!pressure) {
    co_return;
  }
  const rund::task::Result<void> child =
      co_await CompleteIndex(unexpected_child);
  child_rejected->store(
      !child && child.code() == rund::ReasonCode::TaskCapacityExceeded,
      std::memory_order_relaxed);
}


}  // namespace

int CheckContinuations() {
  constexpr std::uint32_t kYieldRounds = 32u;
  std::uint32_t yield_segments = 0u;
  bool yield_overflow_rejected = false;
  rund::task::Status yield_joined{};
  const rund::Session::Result yield_report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler = {
              .task_workers = 1u,
              .task_capacity = 2u,
              .ready_queue_capacity = 1u,
          },
      },
      [&] {
        const rund::task::Handle yielding = rund::task::spawn(
            "ready-wake-once", YieldRounds(&yield_segments, kYieldRounds));
        const rund::task::Handle overflow =
            rund::task::spawn("ready-admission-overflow", [] {});
        yield_overflow_rejected =
            !overflow && overflow.code() ==
                             rund::ReasonCode::ReadyQueueCapacityExceeded;
        yield_joined = rund::task::join(yielding);
      });
  TEST_ASSERT(yield_report.ok());
  TEST_ASSERT(yield_joined.ok());
  TEST_ASSERT(yield_overflow_rejected);
  TEST_ASSERT(yield_segments == kYieldRounds + 1u);
  TEST_ASSERT(yield_report.tasks().spawned() == 1u);
  TEST_ASSERT(yield_report.tasks().yields() == kYieldRounds);
  TEST_ASSERT(yield_report.tasks().coroutine_resumes() == kYieldRounds + 1u);
  TEST_ASSERT(yield_report.tasks().max_ready_depth() == 1u);
  TEST_ASSERT(yield_report.tasks().global_ready_queue_pushes() ==
              kYieldRounds + 1u);
  TEST_ASSERT(yield_report.tasks().global_ready_queue_pops() ==
              kYieldRounds + 1u);
  TEST_ASSERT(yield_report.tasks().ready_spawn_pushes() == 1u);
  TEST_ASSERT(yield_report.tasks().ready_progress_pushes() == kYieldRounds);

  std::atomic<std::uint64_t> queued{0u};
  std::atomic<std::uint64_t> continued{0u};
  std::atomic<bool> continuation_overflow_rejected{false};
  rund::task::Status continuation_joined{};
  const rund::Session::Result continuation_report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler = {
              .task_workers = 1u,
              .task_capacity = 3u,
              .ready_queue_capacity = 1u,
          },
      },
      [&] {
        const rund::task::Handle parent = rund::task::spawn(
            "ready-continuation-parent",
            AwaitAtReadyLimit(&queued, &continued,
                              &continuation_overflow_rejected));
        continuation_joined = rund::task::join(parent);
      });
  TEST_ASSERT(continuation_report.ok());
  TEST_ASSERT(continuation_joined.ok());
  TEST_ASSERT(continuation_overflow_rejected.load(std::memory_order_relaxed));
  TEST_ASSERT(queued.load(std::memory_order_relaxed) == 1u);
  TEST_ASSERT(continued.load(std::memory_order_relaxed) == 1u);
  TEST_ASSERT(continuation_report.tasks().spawned() == 3u);
  TEST_ASSERT(continuation_report.tasks().completed() == 3u);
  TEST_ASSERT(continuation_report.tasks().failed() == 0u);
  TEST_ASSERT(continuation_report.tasks().max_ready_depth() == 2u);

  std::atomic<std::uint64_t> unexpected_child{0u};
  std::atomic<bool> continuation_task_limit_rejected{false};
  rund::task::Status task_limit_joined{};
  const rund::Session::Result task_limit_report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler = {
              .task_workers = 1u,
              .task_capacity = 2u,
              .ready_queue_capacity = 2u,
          },
      },
      [&] {
        const rund::task::Handle parent = rund::task::spawn(
            "task-capacity-parent",
            AwaitAtTaskLimit(&unexpected_child,
                             &continuation_task_limit_rejected));
        task_limit_joined = rund::task::join(parent);
      });
  TEST_ASSERT(task_limit_report.ok());
  TEST_ASSERT(task_limit_joined.ok());
  TEST_ASSERT(
      continuation_task_limit_rejected.load(std::memory_order_relaxed));
  TEST_ASSERT(unexpected_child.load(std::memory_order_relaxed) == 0u);
  TEST_ASSERT(task_limit_report.tasks().spawned() == 2u);
  TEST_ASSERT(task_limit_report.tasks().completed() == 2u);
  TEST_ASSERT(task_limit_report.tasks().failed() == 0u);

  bool unrelated_completed = false;
  bool scoped_timer_completed = false;
  rund::task::Status scoped_timer{};
  rund::task::Status unrelated_joined{};
  const rund::Session::Result scoped_timer_report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler = {
              .task_workers = 1u,
              .task_capacity = 2u,
              .ready_queue_capacity = 2u,
              .timer_capacity = 1u,
          },
      },
      [&] {
        const rund::task::Handle unrelated = rund::task::spawn(
            "scope-unrelated-ready", [&] { unrelated_completed = true; });
        scoped_timer = rund::task::scope([&] {
          TEST_ASSERT(rund::task::spawn(
              "scope-timer", CompleteAfterTimer(&scoped_timer_completed)));
        });
        unrelated_joined = rund::task::join(unrelated);
      });
  TEST_ASSERT(scoped_timer_report);
  TEST_ASSERT(scoped_timer);
  TEST_ASSERT(unrelated_joined);
  TEST_ASSERT(scoped_timer_completed);
  TEST_ASSERT(unrelated_completed);

  return 0;
}

}  // namespace ready_queue_detail
