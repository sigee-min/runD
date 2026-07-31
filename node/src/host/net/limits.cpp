#include <rund/net/limits.hpp>

#include "../../runtime/task/scheduler/access.hpp"
#include "../../runtime/task/scheduler/state.hpp"

namespace rund::net {

Limits limits() noexcept {
  node::Scheduler *const scheduler = node::scheduler_access::ActiveScheduler();
  if (scheduler == nullptr) {
    return Limits{::rund::ReasonCode::NodeRuntimeMissing};
  }
  return scheduler->ReadLimits();
}

} // namespace rund::net
