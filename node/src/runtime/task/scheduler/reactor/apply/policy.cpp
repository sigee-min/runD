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
    : reactor(&runtime),
      active(true),
      batch_add_defer(true) {
  ++reactor->apply_policy.defer_depth;
  if (batch_add_defer) {
    ++reactor->apply_policy.batch_add_defer_depth;
  }
  reactor->apply_policy.defer_registration_apply = true;
}

ReactorApplyBatchScope::~ReactorApplyBatchScope() {
  if (!active || reactor == nullptr ||
      reactor->apply_policy.defer_depth == 0u) {
    return;
  }
  --reactor->apply_policy.defer_depth;
  if (batch_add_defer &&
      reactor->apply_policy.batch_add_defer_depth != 0u) {
    --reactor->apply_policy.batch_add_defer_depth;
  }
  if (reactor->apply_policy.defer_depth == 0u) {
    reactor->apply_policy.defer_registration_apply = false;
  }
}

bool ReactorApplyPolicyShouldDefer(
    ReactorRuntime& reactor,
    const std::size_t ready_depth,
    const bool force) noexcept {
  if (force || reactor.changes.empty() ||
      !reactor.apply_policy.defer_registration_apply) {
    return false;
  }
  if (ready_depth == 0u &&
      (reactor.apply_policy.batch_add_defer_depth == 0u ||
       !PendingChangesAreAddsOnly(reactor))) {
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
  const bool deferred_scope_active =
      reactor.apply_policy.defer_registration_apply ||
      reactor.apply_policy.defer_depth != 0u;
  if (deferred_scope_active) {
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
