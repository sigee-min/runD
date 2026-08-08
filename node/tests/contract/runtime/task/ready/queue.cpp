#include "test/assert.hpp"

#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>
#include <rund/task/channel.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "../../../../../src/runtime/task/scheduler/progress/ready/pick.hpp"
#include "../../../../../src/runtime/task/scheduler/state.hpp"
#include "../../../../../src/runtime/task/scheduler/state/model/work.hpp"
#include "../../../../../src/runtime/task/scheduler/state/storage/ready/queue.hpp"

namespace {

rund::task::Task<void> HoldIndex(
    rund::task::channel<std::uint32_t>* const gate) {
  (void)co_await gate->recv();
}

rund::task::Task<void> CompleteIndex(
    std::atomic<std::uint64_t>* const completed) {
  completed->fetch_add(1u, std::memory_order_relaxed);
  co_return;
}

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

struct LaneGate final {
  std::atomic<bool> started{false};
  std::atomic<bool> release{false};
};

void BlockLane(void *const context) noexcept {
  auto &gate = *static_cast<LaneGate *>(context);
  gate.started.store(true, std::memory_order_release);
  gate.started.notify_one();
  while (!gate.release.load(std::memory_order_acquire)) {
    gate.release.wait(false, std::memory_order_acquire);
  }
}

struct LeafOrder final {
  std::uint64_t trace = 0u;
  bool second_calculated_first = false;
};

LeafOrder RunLeafOrder(const bool blocked) {
  LaneGate gate{};
  rund::node::SchedulerWork work{
      .context = &gate,
      .invoke = BlockLane,
  };
  std::atomic<bool> first_calculated{false};
  std::atomic<bool> second_calculated_first{false};
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .id = 91u,
          .workers = 1u,
          .scheduler = {
              .task_workers = 2u,
              .task_capacity = 2u,
              .ready_queue_capacity = 2u,
          },
      },
      [&] {
        if (blocked) {
          rund::node::Scheduler *const scheduler =
              rund::node::Scheduler::Active();
          TEST_ASSERT(scheduler != nullptr);
          TEST_ASSERT(scheduler->EnqueueWork(&work, 0u));
          gate.started.wait(false, std::memory_order_acquire);
        }
        std::array<rund::task::Handle, 2u> handles{
            rund::task::spawn("ordered-first", [&] {
              first_calculated.store(true, std::memory_order_release);
            }),
            rund::task::spawn("ordered-second", [&] {
              second_calculated_first.store(
                  !first_calculated.load(std::memory_order_acquire),
                  std::memory_order_release);
              gate.release.store(true, std::memory_order_release);
              gate.release.notify_one();
            }),
        };
        TEST_ASSERT(handles[0]);
        TEST_ASSERT(handles[1]);
        joined = rund::task::join_all(handles);
      });
  TEST_ASSERT(report);
  TEST_ASSERT(joined);
  TEST_ASSERT(first_calculated.load(std::memory_order_acquire));
  return LeafOrder{
      .trace = report.tasks().trace_hash(),
      .second_calculated_first =
          second_calculated_first.load(std::memory_order_acquire),
  };
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

int RunRuntimeTaskReadyQueueContract() {
  using rund::node::ReadyPick;
  using rund::node::ReadyPickDisposition;

  static_assert(!std::is_aggregate_v<ReadyPick>);
  static_assert(!std::is_default_constructible_v<ReadyPick>);
  static_assert(std::is_trivially_copyable_v<ReadyPick>);
  constexpr ReadyPick no_pick = ReadyPick::none();
  static_assert(no_pick.disposition() == ReadyPickDisposition::None);
  static_assert(no_pick.task_id() == 0u);
  constexpr ReadyPick zero_task = ReadyPick::task(0u);
  static_assert(zero_task.disposition() == ReadyPickDisposition::None);
  static_assert(zero_task.task_id() == 0u);
  constexpr ReadyPick task_pick = ReadyPick::task(17u);
  static_assert(task_pick.disposition() == ReadyPickDisposition::Task);
  static_assert(task_pick.task_id() == 17u);
  constexpr ReadyPick blocked_pick = ReadyPick::blocked();
  static_assert(blocked_pick.disposition() == ReadyPickDisposition::Blocked);
  static_assert(blocked_pick.task_id() == 0u);
  constexpr ReadyPick activity_pick = ReadyPick::activity();
  static_assert(activity_pick.disposition() == ReadyPickDisposition::Activity);
  static_assert(activity_pick.task_id() == 0u);

  ReadyQueue queue{};
  queue.configure(4u);
  TEST_ASSERT(queue.capacity() == 4u);
  TEST_ASSERT(queue.empty());

  TEST_ASSERT(queue.push_back(1u));
  TEST_ASSERT(queue.push_back(2u));
  TEST_ASSERT(queue.push_back(3u));
  TEST_ASSERT(queue.push_back(4u));
  TEST_ASSERT(!queue.push_back(5u));

  std::uint64_t id = 0u;
  TEST_ASSERT(queue.pop_front(id) && id == 1u);
  TEST_ASSERT(queue.pop_front(id) && id == 2u);
  TEST_ASSERT(queue.push_back(5u));
  TEST_ASSERT(queue.push_back(6u));

  TEST_ASSERT(queue.front(id) && id == 3u);

  TEST_ASSERT(queue.remove(4u));
  TEST_ASSERT(queue.push_front(2u));
  TEST_ASSERT(queue.take_first(
      [](const std::uint64_t candidate) { return candidate == 5u; }, id));
  TEST_ASSERT(id == 5u);

  TEST_ASSERT(queue.pop_front(id) && id == 2u);
  TEST_ASSERT(queue.pop_front(id) && id == 3u);
  TEST_ASSERT(queue.pop_front(id) && id == 6u);
  TEST_ASSERT(queue.empty());
  TEST_ASSERT(!queue.front(id));
  TEST_ASSERT(!queue.pop_front(id));

  const LeafOrder ordinary_order = RunLeafOrder(false);
  const LeafOrder blocked_order = RunLeafOrder(true);
  TEST_ASSERT(blocked_order.second_calculated_first);
  TEST_ASSERT(blocked_order.trace == ordinary_order.trace);

  queue.clear();
  TEST_ASSERT(queue.push_front(9u));
  TEST_ASSERT(queue.pop_front(id) && id == 9u);

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

  constexpr std::size_t kRounds = 64u;
  constexpr std::size_t kTasks = 4u;
  std::uint64_t completed = 0u;
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
        .workers = 1u,
        .scheduler = {
          .task_workers = 1u,
          .task_capacity = 4u,
          .ready_queue_capacity = 4u,
        },
      },
      [&] {
        std::array<rund::task::Handle, kTasks> handles{};
        for (std::size_t round = 0u; round < kRounds; ++round) {
          for (auto& handle : handles) {
            handle = rund::task::spawn("index-reuse", [&completed] {
              ++completed;
            });
            TEST_ASSERT(handle);
          }
          TEST_ASSERT(rund::task::join_all(handles));
        }
      });
  TEST_ASSERT(report);
  TEST_ASSERT(completed == kRounds * kTasks);

  completed = 0u;
  const rund::Session::Result wrap_report = rund::run(
      rund::SessionConfig{
        .workers = 1u,
        .scheduler = {
          .task_workers = 1u,
          .task_capacity = 4u,
          .ready_queue_capacity = 4u,
          .channel_capacity = 1u,
          .channel_wait_capacity = 4u,
        },
      },
      [&] {
        auto gate = rund::task::channel<std::uint32_t>::make(0u);
        const auto held = rund::task::spawn("index-held", HoldIndex(&gate));
        TEST_ASSERT(held);
        std::array<rund::task::Handle, 3u> handles{};
        for (std::size_t round = 0u; round < kRounds; ++round) {
          for (auto& handle : handles) {
            handle = rund::task::spawn("index-wrap", [&completed] {
              ++completed;
            });
            TEST_ASSERT(handle);
          }
          TEST_ASSERT(rund::task::join_all(handles));
        }
        TEST_ASSERT(gate.close());
        TEST_ASSERT(rund::task::join(held));
      });
  TEST_ASSERT(wrap_report);
  TEST_ASSERT(completed == kRounds * 3u);

  constexpr std::size_t kBatchTasks = 4096u;
  std::atomic<std::uint64_t> batch_completed{0u};
  const rund::Session::Result batch_report = rund::run(
      rund::SessionConfig{
        .workers = 1u,
        .scheduler = {
          .task_workers = 4u,
          .task_capacity = kBatchTasks,
          .ready_queue_capacity = kBatchTasks,
        },
      },
      [&] {
        std::vector<rund::task::Handle> handles{};
        handles.reserve(kBatchTasks);
        for (std::size_t index = 0u; index < kBatchTasks; ++index) {
          handles.push_back(rund::task::spawn("index-batch", [&] {
            batch_completed.fetch_add(1u, std::memory_order_relaxed);
          }));
          TEST_ASSERT(handles.back());
        }
        TEST_ASSERT(rund::task::join_all(handles));
      });
  TEST_ASSERT(batch_report);
  TEST_ASSERT(batch_completed.load(std::memory_order_relaxed) == kBatchTasks);
  TEST_ASSERT(batch_report.tasks().spawned() == kBatchTasks);
  TEST_ASSERT(batch_report.tasks().completed() == kBatchTasks);
  TEST_ASSERT(batch_report.tasks().failed() == 0u);
  TEST_ASSERT(batch_report.tasks().callable_resets() == kBatchTasks);
  TEST_ASSERT(batch_report.tasks().participating_task_workers() == 4u);
  TEST_ASSERT(batch_report.tasks().lane_dispatch_batch_packets() == 1u);
  TEST_ASSERT(batch_report.tasks().lane_dispatch_batch_logical_tasks() ==
              kBatchTasks);
  TEST_ASSERT(batch_report.tasks().lane_dispatch_batch_scratch_reuses() >= 1u);

  std::atomic<std::uint64_t> mixed_completed{0u};
  const rund::Session::Result mixed_report = rund::run(
      rund::SessionConfig{
        .workers = 1u,
        .scheduler = {
          .task_workers = 4u,
          .task_capacity = 8u,
          .ready_queue_capacity = 8u,
        },
      },
      [&] {
        std::array<rund::task::Handle, 8u> handles{};
        for (std::size_t index = 0u; index < handles.size(); ++index) {
          if ((index & 1u) == 0u) {
            handles[index] = rund::task::spawn("mixed-leaf", [&] {
              mixed_completed.fetch_add(1u, std::memory_order_relaxed);
            });
          } else {
            handles[index] = rund::task::spawn(
                "mixed-coroutine", CompleteIndex(&mixed_completed));
          }
          TEST_ASSERT(handles[index]);
        }
        TEST_ASSERT(rund::task::join_all(handles));
      });
  TEST_ASSERT(mixed_report);
  TEST_ASSERT(mixed_completed.load(std::memory_order_relaxed) == 8u);
  TEST_ASSERT(mixed_report.tasks().completed() == 8u);
  TEST_ASSERT(mixed_report.tasks().failed() == 0u);

  for (const std::uint32_t workers : {1u, 4u}) {
    rund::task::Status failure{};
    const rund::Session::Result failure_report = rund::run(
        rund::SessionConfig{
          .workers = 1u,
          .scheduler = {
            .task_workers = workers,
            .task_capacity = 8u,
            .ready_queue_capacity = 8u,
          },
        },
        [&] {
          std::array<rund::task::Handle, 8u> handles{};
          for (std::size_t index = 0u; index < handles.size(); ++index) {
            handles[index] = rund::task::spawn("ordered-failure", [index] {
              if (index == 2u) {
                (void)rund::task::join(rund::task::Handle{});
              } else if (index == 5u) {
                throw 1;
              }
            });
            TEST_ASSERT(handles[index]);
          }
          failure = rund::task::join_all(handles);
        });
    TEST_ASSERT(failure_report);
    TEST_ASSERT(!failure);
    TEST_ASSERT(failure.code() ==
                rund::ReasonCode::TaskLeafPrimitiveForbidden);
  }
  return 0;
}
