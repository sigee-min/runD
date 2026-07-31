#include "../local.hpp"

rund::SessionConfig NetServerRunSpec() noexcept {
  return rund::SessionConfig{
      .workers = 1u,
      .scheduler =
          {
              .task_capacity = 8u,
              .ready_queue_capacity = 8u,
              .reactor_wait_capacity = 8u,
              .observation_capacity = 64u,
              .host_event_capacity = 64u,
          },
  };
}
