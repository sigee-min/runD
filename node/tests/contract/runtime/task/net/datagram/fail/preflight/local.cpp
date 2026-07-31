#include "local.hpp"

rund::SessionConfig DatagramPreflightRunSpec() noexcept {
  return rund::SessionConfig{
      .workers = 1u,
      .scheduler =
          {
              .task_workers = 1u,
              .task_capacity = 2u,
              .ready_queue_capacity = 4u,
              .timer_capacity = 4u,
              .reactor_wait_capacity = 4u,
              .net_datagram_capacity_bytes = 8u,
              .host_event_capacity = 8u,
          },
  };
}
