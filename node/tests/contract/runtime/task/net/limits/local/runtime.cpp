#include "../local.hpp"

namespace rund::node::test_contract::net_limits {

rund::SessionConfig Config() noexcept {
  rund::SessionConfig options{};
  options.id = 987u;
  options.workers = 1u;
  options.trace_capacity = 256u;
  options.scheduler.task_capacity = 8u;
  options.scheduler.ready_queue_capacity = 16u;
  options.scheduler.task_workers = 1u;
  options.scheduler.timer_capacity = 8u;
  options.scheduler.reactor_wait_capacity = 32u;
  options.scheduler.observation_capacity = 128u;
  options.scheduler.host_event_capacity = 128u;
  return options;
}

} // namespace rund::node::test_contract::net_limits
