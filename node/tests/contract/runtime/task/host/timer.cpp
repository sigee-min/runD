#include "test/assert.hpp"

#include <rund/host.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include <limits>

static_assert((rund::host::chrono::time_point{
                   .ns = std::numeric_limits<std::int64_t>::max()} -
               rund::host::chrono::time_point{
                   .ns = std::numeric_limits<std::int64_t>::min()})
                  .count() == std::numeric_limits<std::int64_t>::max());
static_assert((rund::host::chrono::time_point{
                   .ns = std::numeric_limits<std::int64_t>::min()} -
               rund::host::chrono::time_point{
                   .ns = std::numeric_limits<std::int64_t>::max()})
                  .count() == std::numeric_limits<std::int64_t>::min());

int RunRuntimeTaskHostTimerContract() {
  rund::task::Status at_sleep{};
  rund::task::Status joined{};

  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 2u,
                  .ready_queue_capacity = 2u,
                  .timer_capacity = 2u,
                  .observation_capacity = 2u,
              },
      },
      [&] {
        const rund::task::Handle task =
            rund::task::spawn("host-timer-at-zero", [&] {
              const rund::task::SleepOp sleep = rund::host::timer::at(
                  rund::host::chrono::time_point{.ns = 0});
              at_sleep = sleep ? rund::task::Status::success()
                               : rund::task::Status::fail(sleep.code());
            });
        joined = rund::task::join(task);
      });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(at_sleep.ok());
  return 0;
}
