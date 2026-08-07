#include "cancel/local.hpp"

namespace rund::task {

stop_source stop_source::create() noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    return {};
  }
  const ::rund::detail::task::StopIdentity identity =
      scheduler->CreateStopSource();
  if (!identity) {
    return {};
  }
  return stop_source{stop_token{identity}};
}

Status stop_source::request_stop() const noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    return Status::fail(ReasonCode::NodeRuntimeMissing);
  }
  return scheduler->RequestStop(token_.identity_);
}

StopState stop_token::state() const noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    return StopState::fail(ReasonCode::NodeRuntimeMissing);
  }
  return scheduler->StopRequested(identity_);
}

}  // namespace rund::task
