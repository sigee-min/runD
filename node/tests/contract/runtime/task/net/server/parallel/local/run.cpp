#include "../local.hpp"

rund::SessionConfig NetServerParallelRunSpec() noexcept {
  return rund::SessionConfig{
      .workers = 4u,
      .scheduler =
          {
              .task_capacity = 9u,
              .ready_queue_capacity = 9u,
              .reactor_wait_capacity = 8u,
              .observation_capacity = 64u,
              .host_event_capacity = 64u,
          },
  };
}
