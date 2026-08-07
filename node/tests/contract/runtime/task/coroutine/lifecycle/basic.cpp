#include <rund/task/api.hpp>
#include <rund/task/await.hpp>
#include "../local.hpp"

namespace rund::node::test_contract::coroutine {

rund::task::Task<void>
CompleteOnWorker(std::atomic<std::uint32_t> *const body_runs,
                 std::thread::id *const body_thread) {
  *body_thread = std::this_thread::get_id();
  body_runs->fetch_add(1u, std::memory_order_release);
  co_return;
}

rund::task::Task<void>
YieldOnce(std::atomic<std::uint32_t> *const after_await) {
  const rund::task::Status yielded = co_await rund::task::yield();
  (void)yielded;
  after_await->fetch_add(1u, std::memory_order_release);
}

rund::task::Task<void>
SleepTwice(std::atomic<std::uint32_t> *const after_await) {
  for (std::uint32_t index = 0u; index < 2u; ++index) {
    const rund::task::Status slept =
        co_await rund::task::sleep(std::chrono::nanoseconds{1});
    (void)slept;
  }
  after_await->fetch_add(1u, std::memory_order_release);
}

rund::task::Task<void>
DiscardOperations(std::atomic<std::uint32_t> *const completed,
                  const rund::host::io::FdView read_fd) {
  (void)rund::task::yield();
  (void)rund::task::sleep(std::chrono::nanoseconds{1});
  (void)::rund::host::io::readable(read_fd);
  completed->fetch_add(1u, std::memory_order_release);
  co_return;
}

int CheckCoroutineComplete() {
  std::atomic<std::uint32_t> body_runs{0u};
  std::thread::id root_thread{};
  std::thread::id body_thread{};
  bool handle_valid = false;
  rund::task::Status task_join{};

  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .id = 785u,
          .workers = 2u,
          .scheduler =
              {
                  .task_capacity = 4u,
                  .ready_queue_capacity = 4u,
              },
      },
      [&] {
        root_thread = std::this_thread::get_id();
        const rund::task::Handle task = rund::task::spawn(
            "coroutine-complete", CompleteOnWorker(&body_runs, &body_thread));
        handle_valid = static_cast<bool>(task);
        task_join = rund::task::join(task);
      });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(handle_valid);
  TEST_ASSERT(task_join.ok());
  TEST_ASSERT(task_join.code() == rund::ReasonCode::Ok);
  TEST_ASSERT(body_runs.load(std::memory_order_acquire) == 1u);
  TEST_ASSERT(body_thread != std::thread::id{});
  TEST_ASSERT(body_thread != root_thread);
  TEST_ASSERT(report.tasks().spawned() == 1u);
  TEST_ASSERT(report.tasks().completed() == 1u);
  TEST_ASSERT(report.tasks().failed() == 0u);
  TEST_ASSERT(report.tasks().coroutine_tasks_admitted() == 1u);
  TEST_ASSERT(report.tasks().coroutine_resumes() == 1u);
  TEST_ASSERT(report.tasks().coroutine_completions() == 1u);
  TEST_ASSERT(report.tasks().coroutine_failures() == 0u);
  TEST_ASSERT(report.tasks().coroutine_frame_destroys() == 1u);
  return 0;
}

int CheckCoroutineYield() {
  std::atomic<std::uint32_t> after_yield{0u};
  rund::task::Status joined{};
  bool handle_valid = false;
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .id = 786u,
          .workers = 2u,
          .scheduler =
              {
                  .task_capacity = 4u,
                  .ready_queue_capacity = 4u,
              },
      },
      [&] {
        const rund::task::Handle task =
            rund::task::spawn("coroutine-yield-once", YieldOnce(&after_yield));
        handle_valid = static_cast<bool>(task);
        joined = rund::task::join(task);
      });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(handle_valid);
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(joined.code() == rund::ReasonCode::Ok);
  TEST_ASSERT(after_yield.load(std::memory_order_acquire) == 1u);
  TEST_ASSERT(report.tasks().spawned() == 1u);
  TEST_ASSERT(report.tasks().completed() == 1u);
  TEST_ASSERT(report.tasks().failed() == 0u);
  TEST_ASSERT(report.tasks().yields() == 1u);
  TEST_ASSERT(report.tasks().parked() == 1u);
  TEST_ASSERT(report.tasks().coroutine_tasks_admitted() == 1u);
  TEST_ASSERT(report.tasks().coroutine_resumes() == 2u);
  TEST_ASSERT(report.tasks().coroutine_parks() == 1u);
  TEST_ASSERT(report.tasks().coroutine_wakes() == 1u);
  TEST_ASSERT(report.tasks().coroutine_completions() == 1u);
  TEST_ASSERT(report.tasks().coroutine_failures() == 0u);
  TEST_ASSERT(report.tasks().coroutine_frame_destroys() == 1u);
  return 0;
}

int CheckCoroutineSleep() {
  std::atomic<std::uint32_t> after_sleep{0u};
  rund::task::Status joined{};
  bool handle_valid = false;
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .id = 787u,
          .workers = 2u,
          .scheduler =
              {
                  .task_capacity = 4u,
                  .ready_queue_capacity = 4u,
                  .observation_capacity = 1u,
              },
      },
      [&] {
        const rund::task::Handle task =
            rund::task::spawn("coroutine-sleep-twice", SleepTwice(&after_sleep));
        handle_valid = static_cast<bool>(task);
        joined = rund::task::join(task);
      });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(handle_valid);
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(joined.code() == rund::ReasonCode::Ok);
  TEST_ASSERT(after_sleep.load(std::memory_order_acquire) == 1u);
  TEST_ASSERT(report.tasks().spawned() == 1u);
  TEST_ASSERT(report.tasks().completed() == 1u);
  TEST_ASSERT(report.tasks().failed() == 0u);
  TEST_ASSERT(report.tasks().timers() == 2u);
  TEST_ASSERT(report.tasks().observations() == 2u);
  TEST_ASSERT(report.observations().size() == 1u);
  TEST_ASSERT(report.tasks().observation_dropped() == 1u);
  TEST_ASSERT(report.tasks().parked() == 2u);
  TEST_ASSERT(report.tasks().coroutine_tasks_admitted() == 1u);
  TEST_ASSERT(report.tasks().coroutine_resumes() == 3u);
  TEST_ASSERT(report.tasks().coroutine_parks() == 2u);
  TEST_ASSERT(report.tasks().coroutine_wakes() == 2u);
  TEST_ASSERT(report.tasks().coroutine_completions() == 1u);
  TEST_ASSERT(report.tasks().coroutine_failures() == 0u);
  TEST_ASSERT(report.tasks().coroutine_frame_destroys() == 1u);
  return 0;
}

int CheckDiscardedOperations() {
  std::atomic<std::uint32_t> completed{0u};
  int pipe_fds[2] = {-1, -1};
  TEST_ASSERT(::pipe(pipe_fds) == 0);
  rund::host::io::Fd ready_fd =
      rund::host::io::take_native_fd(::dup(pipe_fds[0]));
  TEST_ASSERT(ready_fd);
  bool handle_ok = false;
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .id = 814u,
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 4u,
                  .ready_queue_capacity = 4u,
              },
      },
      [&] {
        const rund::task::Handle task =
            rund::task::spawn("coroutine-discard-operations",
                              DiscardOperations(&completed, ready_fd.view()));
        handle_ok = static_cast<bool>(task);
        if (task) {
          handle_ok = handle_ok && static_cast<bool>(rund::task::join(task));
        }
      });
  TEST_ASSERT(::close(pipe_fds[0]) == 0);
  TEST_ASSERT(::close(pipe_fds[1]) == 0);
  TEST_ASSERT(report.ok());
  TEST_ASSERT(handle_ok);
  TEST_ASSERT(completed.load(std::memory_order_acquire) == 1u);
  TEST_ASSERT(report.tasks().yields() == 0u);
  TEST_ASSERT(report.tasks().timers() == 0u);
  TEST_ASSERT(report.tasks().reactor_waits() == 0u);
  TEST_ASSERT(report.tasks().parked() == 0u);
  return 0;
}

} // namespace rund::node::test_contract::coroutine
