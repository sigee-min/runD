#include <rund/host/chrono.hpp>

#include "../runtime/task/scheduler/host.hpp"

namespace rund::host::chrono {

time_point logical_clock::now() noexcept {
  return time_point{.ns = ::rund::node::scheduler_host::LogicalTimeNs()};
}

} // namespace rund::host::chrono
