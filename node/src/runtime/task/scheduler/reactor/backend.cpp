#include "backend.hpp"

#include "../../../reactor/platform.hpp"
#include "apply/policy.hpp"
#include "backlog.hpp"
#include "change/queue.hpp"
#include "registration.hpp"
#include "scratch.hpp"
#include "stats.hpp"

namespace rund::node {
namespace {

[[nodiscard]] ReactorApplyResult
EnsureBackendOpen(ReactorRuntime &reactor) noexcept {
  const ReactorPlatformOpResult opened = OpenReactorPlatform(reactor.platform);
  switch (opened.disposition()) {
  case ReactorPlatformOpDisposition::Success:
    return ReactorApplyResult::success();
  case ReactorPlatformOpDisposition::BackendUnavailable:
    return ReactorApplyResult::backend_unavailable();
  case ReactorPlatformOpDisposition::Invalid:
  case ReactorPlatformOpDisposition::Failed:
    return ReactorApplyResult::failed();
  }
}

} // namespace

ReactorApplyResult
ReactorBackendApplyChanges(ReactorRuntime &reactor,
                           ::rund::detail::task::StatStorage &stats) noexcept {
  const ReactorApplyResult opened = EnsureBackendOpen(reactor);
  if (opened.disposition() != ReactorApplyDisposition::Success) {
    return opened;
  }
  return ReactorChangeQueueApply(reactor, stats);
}

void ReactorCloseRuntime(ReactorRuntime &reactor) noexcept {
  CloseReactorPlatform(reactor.platform);
  ReactorRegistrationClear(reactor);
  ReactorApplyPolicyClear(reactor);
  ReactorBacklogClear(reactor);
  reactor.changes.clear();
  ReactorScratchClear(reactor);
}

} // namespace rund::node
