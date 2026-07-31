#include "../local.hpp"

rund::SessionConfig TimedReadySuspendRunSpec() noexcept {
  return rund::SessionConfig{
      .workers = 1u,
      .scheduler =
          {
              .task_workers = 1u,
              .task_capacity = 6u,
              .ready_queue_capacity = 6u,
              .timer_capacity = 6u,
              .reactor_wait_capacity = 6u,
              .observation_capacity = 64u,
              .host_event_capacity = 64u,
          },
  };
}
