#include "../../session/status.hpp"
#include "../local.hpp"

namespace rund::node::runtime_detail {

bool Runnable(const ::rund::SessionState state) {
  return state == ::rund::SessionState::Configured ||
         state == ::rund::SessionState::Running;
}

::rund::SessionState ObserveLifecycle(const Runtime &runtime) {
  const auto &state = RuntimeAccess::state(runtime);
  std::lock_guard lock{state.mutex};
  return state.lifecycle;
}

::rund::Session::Status LifecycleFail(const ReasonCode code,
                                      const ::rund::SessionState state) {
  return ::rund::detail::session::StatusAccess::make(code, state);
}

::rund::Session::Status LifecyclePass(const ::rund::SessionState state) {
  return ::rund::detail::session::StatusAccess::make(ReasonCode::Ok, state);
}

} // namespace rund::node::runtime_detail
