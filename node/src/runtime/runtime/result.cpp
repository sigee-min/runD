#include "local.hpp"

#include "../compute/state.hpp"

#include <limits>

namespace rund::node {

using runtime_detail::RuntimeActive;

::rund::Session::Snapshot Runtime::State::SnapshotLocked() const {
  ::rund::Session::Snapshot snapshot{
      .state = lifecycle,
      .scope_active = scope_active,
  };
  if (compute_host != nullptr) {
    std::lock_guard host_lock{compute_host->mutex};
    snapshot.active_compute_jobs = static_cast<std::uint32_t>(
        std::min<std::size_t>(compute_host->outstanding,
                              std::numeric_limits<std::uint32_t>::max()));
  }
  return snapshot;
}

::rund::Session::Snapshot Runtime::snapshot() const {
  const bool reentry = RuntimeActive(this);
  std::lock_guard<std::mutex> lock(state_->mutex);
  ::rund::Session::Snapshot snapshot = state_->SnapshotLocked();
  if (reentry) {
    snapshot.code = ReasonCode::RuntimeReentryForbidden;
  }
  return snapshot;
}

::rund::Trace Runtime::trace() const {
  if (RuntimeActive(this)) {
    return ::rund::Trace{.code = ReasonCode::RuntimeReentryForbidden};
  }
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->TraceLocked();
}

::rund::Trace runtime_detail::RuntimeAccess::take_trace(Runtime &runtime) {
  if (RuntimeActive(&runtime)) {
    return ::rund::Trace{.code = ReasonCode::RuntimeReentryForbidden};
  }
  std::lock_guard<std::mutex> lock(runtime.state_->mutex);
  return runtime.state_->TakeTraceLocked();
}

::rund::Resources Runtime::resources() const {
  if (RuntimeActive(this)) {
    return ::rund::Resources{.code = ReasonCode::RuntimeReentryForbidden};
  }
  std::lock_guard<std::mutex> lock(state_->mutex);
  if (state_->lifecycle == ::rund::SessionState::Unconfigured) {
    return ::rund::Resources{.code = ReasonCode::NotConfigured};
  }
  return state_->resources.observed;
}

} // namespace rund::node
