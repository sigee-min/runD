#include "../local.hpp"

namespace rund::node::test_contract::net_accept_handoff {

rund::SessionConfig RunSpec() noexcept {
  return rund::SessionConfig{
      .workers = 1u,
      .scheduler =
          {
              .task_capacity = 4u,
              .ready_queue_capacity = 4u,
              .reactor_wait_capacity = 4u,
              .observation_capacity = 32u,
              .host_event_capacity = 32u,
          },
  };
}

} // namespace rund::node::test_contract::net_accept_handoff
