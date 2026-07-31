#include "../local.hpp"

rund::SessionConfig NetMultiClientRunSpec() noexcept {
  return rund::SessionConfig{
      .workers = 1u,
      .scheduler =
          {
              .task_capacity = 96u,
              .ready_queue_capacity = 128u,
              .timer_capacity = 128u,
              .reactor_wait_capacity = 128u,
              .observation_capacity = 512u,
              .host_event_capacity = 512u,
          },
  };
}
