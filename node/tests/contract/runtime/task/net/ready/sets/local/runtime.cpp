#include "../local.hpp"

namespace rund::node::test_contract::ready_sets {

rund::SessionConfig
RunSpec(const std::uint32_t task_capacity, const std::uint32_t timer_capacity,
        const std::uint32_t reactor_wait_capacity) noexcept {
  return rund::SessionConfig{
      .workers = 1u,
      .scheduler =
          {
              .task_workers = 1u,
              .task_capacity = task_capacity,
              .ready_queue_capacity = task_capacity + 4u,
              .timer_capacity = timer_capacity,
              .reactor_wait_capacity = reactor_wait_capacity,
              .observation_capacity = 256u,
              .host_event_capacity = 256u,
          },
  };
}

rund::SessionConfig
Config(const std::uint32_t task_capacity, const std::uint32_t timer_capacity,
       const std::uint32_t reactor_wait_capacity,
       const std::uint32_t net_ready_set_capacity,
       const std::uint32_t net_ready_set_member_capacity) noexcept {
  rund::SessionConfig options{};
  options.id = 931u;
  options.workers = 1u;
  options.trace_capacity = 256u;
  options.scheduler.task_capacity = task_capacity;
  options.scheduler.ready_queue_capacity = task_capacity + 4u;
  options.scheduler.task_workers = 1u;
  options.scheduler.timer_capacity = timer_capacity;
  options.scheduler.reactor_wait_capacity = reactor_wait_capacity;
  options.scheduler.observation_capacity = 256u;
  options.scheduler.host_event_capacity = 256u;
  options.scheduler.net_ready_set_capacity = net_ready_set_capacity;
  options.scheduler.net_ready_set_member_capacity =
      net_ready_set_member_capacity;
  return options;
}

} // namespace rund::node::test_contract::ready_sets
