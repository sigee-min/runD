#include "../scheduler/state/storage.hpp"
#include "../scheduler/state/storage/check.hpp"

namespace rund::node {

bool Scheduler::RecordMemory(
    const ::rund::PreparedMemory &memory) noexcept {
  if (active_scheduler_context != nullptr || !memory.capacity.valid()) {
    return false;
  }
  state_->RequireSequencer();
  state_->resources.prepared_memory = memory;
  return true;
}

} // namespace rund::node

namespace rund {

bool record_memory(const PreparedMemory &memory) noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    return false;
  }
  return scheduler->RecordMemory(memory);
}

} // namespace rund
