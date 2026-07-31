#include "../../compute/state.hpp"
#include "../local.hpp"

namespace rund::node {

using runtime_detail::LifecycleFail;
using runtime_detail::LifecyclePass;
using runtime_detail::ObserveLifecycle;
using runtime_detail::RuntimeActive;
using runtime_detail::RuntimeActiveScope;

::rund::Session::Status Runtime::start() {
  if (RuntimeActive(this)) {
    return LifecycleFail(ReasonCode::RuntimeReentryForbidden,
                         ObserveLifecycle(*this));
  }
  std::lock_guard lock{state_->mutex};
  RuntimeActiveScope active(this);
  if (state_->lifecycle == ::rund::SessionState::Unconfigured) {
    return LifecycleFail(ReasonCode::NotConfigured, state_->lifecycle);
  }
  if (state_->lifecycle != ::rund::SessionState::Configured) {
    return LifecycleFail(ReasonCode::NotRunnable, state_->lifecycle);
  }
  state_->lifecycle = ::rund::SessionState::Running;
  if (state_->compute_host != nullptr) {
    std::lock_guard host_lock{state_->compute_host->mutex};
    state_->compute_host->accepting = true;
    state_->compute_host->reject_reason = compute::Reason::RuntimeNotRunning;
  }
  state_->AddTrace(::rund::TraceEvent::RuntimeStarted,
                   ::rund::TraceCode::runtime(ReasonCode::Ok));
  return LifecyclePass(state_->lifecycle);
}

::rund::Session::Status Runtime::shutdown(const Shutdown intent) {
  if (RuntimeActive(this)) {
    return LifecycleFail(ReasonCode::RuntimeReentryForbidden,
                         ObserveLifecycle(*this));
  }
  RuntimeActiveScope active(this);
  std::unique_lock lock{state_->mutex};
  if (state_->lifecycle == ::rund::SessionState::Unconfigured) {
    return LifecycleFail(ReasonCode::NotConfigured, state_->lifecycle);
  }
  if (state_->lifecycle == ::rund::SessionState::Running) {
    runtime_detail::StopAdmission(state_->compute_host);
    state_->lifecycle = ::rund::SessionState::Draining;
    state_->AddTrace(::rund::TraceEvent::RuntimeDraining,
                     ::rund::TraceCode::runtime(ReasonCode::Ok));
  } else if (intent == Shutdown::Admission ||
             state_->lifecycle != ::rund::SessionState::Draining) {
    if (intent == Shutdown::Terminal &&
        state_->lifecycle == ::rund::SessionState::Stopped) {
      return LifecyclePass(state_->lifecycle);
    }
    return LifecycleFail(ReasonCode::NotRunnable, state_->lifecycle);
  }

  if (intent == Shutdown::Admission) {
    return LifecyclePass(state_->lifecycle);
  }

  state_->scope_drained.wait(lock, [&] { return !state_->scope_active; });
  lock.unlock();
  runtime_detail::CloseHost(state_->compute_host);
  lock.lock();
  if (state_->lifecycle == ::rund::SessionState::Stopped) {
    return LifecyclePass(state_->lifecycle);
  }
  state_->lifecycle = ::rund::SessionState::Stopped;
  state_->AddTrace(::rund::TraceEvent::RuntimeStopped,
                   ::rund::TraceCode::runtime(ReasonCode::Ok));
  return LifecyclePass(state_->lifecycle);
}

} // namespace rund::node
