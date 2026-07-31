#include "../state/storage.hpp"

namespace rund::node {

std::int64_t Scheduler::LogicalTimeNs() const noexcept {
  return state_->identity.logical_time_ns.load(std::memory_order_acquire);
}

}  // namespace rund::node
