#include "test/assert.hpp"

#include "local.hpp"

#include <rund/session.hpp>
#include <rund/task/api.hpp>

int RunReplayHostCapacityContract() {
  rund::task::Status capped_join{};
  const rund::Session::Result capped = rund::run(
      rund::SessionConfig{
          .workers = 2u,
          .scheduler =
              {
                  .task_capacity = 4u,
                  .ready_queue_capacity = 4u,
                  .timer_capacity = 4u,
                  .host_event_capacity = 1u,
              },
      },
      [&] {
        const rund::task::Handle first =
            rund::task::spawn("first-host-event", ReplaySleep());
        const rund::task::Handle second = rund::task::spawn(
            "second-host-event", ReplaySleep(std::chrono::nanoseconds{2}));
        capped_join = rund::task::join(first, second);
      });
  TEST_ASSERT(capped.ok());
  TEST_ASSERT(capped_join);
  TEST_ASSERT(capped.events().size() == 1u);
  TEST_ASSERT(capped.tasks().host_events() == 2u);
  TEST_ASSERT(capped.tasks().host_events_dropped() == 1u);
  return 0;
}
