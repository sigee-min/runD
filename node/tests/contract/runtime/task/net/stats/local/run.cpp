#include "../local.hpp"

rund::SessionConfig NetStatsRunSpec() noexcept {
  return rund::SessionConfig{
      .workers = 1u,
      .scheduler =
          {
              .task_workers = 1u,
              .task_capacity = 8u,
              .ready_queue_capacity = 8u,
              .timer_capacity = 8u,
              .reactor_wait_capacity = 8u,
              .observation_capacity = 128u,
              .host_event_capacity = 128u,
          },
  };
}
