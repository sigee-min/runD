#include "../../compute/state.hpp"
#include "../../task/scheduler/access.hpp"
#include "../local.hpp"

namespace rund::node::runtime_detail {
namespace {

thread_local RuntimeActiveScope *active_scope = nullptr;

} // namespace

bool RuntimeActive(const Runtime *const runtime) {
  for (const RuntimeActiveScope *active = active_scope; active != nullptr;
       active = active->previous) {
    if (active->runtime == runtime) {
      return true;
    }
  }

  Scheduler *const scheduler = scheduler_access::ActiveScheduler();
  if (runtime == nullptr || scheduler == nullptr) {
    return false;
  }
  const auto &state = RuntimeAccess::state(*runtime);
  std::lock_guard lock{state.mutex};
  return state.compute_host != nullptr &&
         scheduler == &state.compute_host->scheduler;
}

RuntimeActiveScope::RuntimeActiveScope(const Runtime *const value)
    : runtime(value), previous(active_scope) {
  active_scope = this;
}

RuntimeActiveScope::~RuntimeActiveScope() { active_scope = previous; }

} // namespace rund::node::runtime_detail
