#include "../../state/storage.hpp"
#include "policy.hpp"

#include "../../../../reactor/diagnostics.hpp"

namespace rund::node {
namespace {

[[nodiscard]] bool PendingChangesAreAddsOnly(
    const ReactorRuntime& reactor) noexcept {
  if (reactor.changes.empty()) {
    return false;
  }
  for (const ReactorRegistrationChange& change : reactor.changes) {
    if (change.kind() != ReactorRegistrationChange::Kind::Add) {
      return false;
    }
  }
  return true;
}

}  // namespace

ReactorApplyBatchScope::ReactorApplyBatchScope(
    ReactorRuntime& runtime) noexcept
    : reactor_(runtime) {
  ++reactor_.apply_policy.batch_scope_depth;
}

ReactorApplyBatchScope::~ReactorApplyBatchScope() {
  if (reactor_.apply_policy.batch_scope_depth == 0u) {
    return;
  }
  --reactor_.apply_policy.batch_scope_depth;
}

bool ReactorApplyPolicyShouldDefer(
    ReactorRuntime& reactor,
    const std::size_t ready_depth,
    const bool force) noexcept {
  if (force || reactor.changes.empty() ||
      reactor.apply_policy.batch_scope_depth == 0u) {
    return false;
  }
  if (ready_depth == 0u && !PendingChangesAreAddsOnly(reactor)) {
    return false;
  }
  RecordReactorRegistrationApplyDeferral();
  return true;
}

void ReactorApplyPolicyRecordFlush(
    ReactorRuntime& reactor,
    const bool forced) noexcept {
  if (reactor.changes.empty()) {
    return;
  }
  if (reactor.apply_policy.batch_scope_depth != 0u) {
    RecordReactorRegistrationApplyDeferredFlush();
  }
  if (forced) {
    RecordReactorRegistrationApplyForcedFlush();
    return;
  }
}

void ReactorApplyPolicyClear(ReactorRuntime& reactor) noexcept {
  reactor.apply_policy = ReactorApplyPolicy{};
}

}  // namespace rund::node
