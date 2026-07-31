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
  if (!opened.ok) {
    return ReactorApplyResult{
        .ok = false,
        .invalid = opened.invalid,
        .unavailable = opened.unavailable,
        .invalid_handle = kInvalidReactorHandle,
    };
  }
  return {};
}

} // namespace

ReactorApplyResult
ReactorBackendApplyChanges(ReactorRuntime &reactor,
                           ::rund::detail::task::StatStorage &stats) noexcept {
  const ReactorApplyResult opened = EnsureBackendOpen(reactor);
  if (!opened.ok) {
    return opened;
  }
  return ReactorChangeQueueApply(reactor, stats);
}

ReactorPlatformPollResult ReactorBackendPoll(
    ReactorRuntime &reactor, ::rund::detail::task::StatStorage &stats,
    const int timeout_ms, const std::size_t max_events) noexcept {
  const ReactorApplyResult applied = ReactorBackendApplyChanges(reactor, stats);
  if (!applied.ok) {
    return ReactorPlatformPollResult{
        .ok = false,
        .invalid = applied.invalid,
        .unavailable = applied.unavailable,
        .ready = nullptr,
    };
  }
  return PollReactorPlatform(reactor.platform, timeout_ms, max_events);
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
