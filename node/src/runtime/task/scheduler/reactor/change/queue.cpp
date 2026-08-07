#include "queue.hpp"

#include <algorithm>
#include <cstddef>

#include "../../../../reactor/diagnostics.hpp"
#include "../../../../reactor/platform.hpp"
#include "../model.hpp"
#include "../stats.hpp"

namespace rund::node {
namespace {

[[nodiscard]] bool
IgnorableInvalidRemove(const ReactorRegistrationChange &change,
                       const ReactorPlatformBatchResult &result) noexcept {
  return change.is_cleanup_remove() &&
         result.disposition() == ReactorPlatformBatchDisposition::Invalid;
}

[[nodiscard]] bool
StrictChangeForHandle(const ReactorRegistrationChange &change,
                      const ReactorHandle handle) noexcept {
  return change.handle() == handle && !change.is_cleanup_remove();
}

void RecordApplied(::rund::detail::task::StatStorage &stats,
                   const std::size_t count) noexcept {
  for (std::size_t index = 0u; index < count; ++index) {
    RecordReactorRegistrationChangeApplied(stats);
  }
}

[[nodiscard]] ReactorApplyResult
ProjectApplyFailure(const ReactorPlatformBatchResult &result,
                    const ReactorRegistrationChange &failed) noexcept {
  switch (result.disposition()) {
  case ReactorPlatformBatchDisposition::Invalid:
    return ReactorApplyResult::invalid(failed.handle(),
                                       failed.fd_generation());
  case ReactorPlatformBatchDisposition::BackendUnavailable:
    return ReactorApplyResult::backend_unavailable();
  case ReactorPlatformBatchDisposition::Failed:
    return ReactorApplyResult::failed();
  case ReactorPlatformBatchDisposition::Success:
    return ReactorApplyResult::success();
  }
}

} // namespace

ReactorApplyResult
ReactorChangeQueueApply(ReactorRuntime &reactor,
                        ::rund::detail::task::StatStorage &stats) noexcept {
  const std::size_t batch_size = reactor.changes.size();
  if (batch_size == 0u) {
    return ReactorApplyResult::success();
  }

  RecordReactorRegistrationApplyCall(stats);
  RecordReactorRegistrationApplyBatch(batch_size);

  std::size_t cursor = 0u;
  while (cursor < reactor.changes.size()) {
    const ReactorRegistrationChange *const first =
        reactor.changes.data() + cursor;
    const std::size_t remaining = reactor.changes.size() - cursor;
    const ReactorPlatformBatchResult result =
        ApplyReactorPlatformChanges(reactor.platform, first, remaining);
    if (result.disposition() == ReactorPlatformBatchDisposition::Success) {
      RecordApplied(stats, remaining);
      reactor.changes.clear();
      return ReactorApplyResult::success();
    }

    const std::size_t failed_offset =
        std::min<std::size_t>(result.failed_index(), remaining - 1u);
    RecordApplied(stats, failed_offset);
    const std::size_t failed_index = cursor + failed_offset;
    const ReactorRegistrationChange failed = reactor.changes[failed_index];
    if (IgnorableInvalidRemove(failed, result)) {
      RecordReactorDeferredRemoveInvalidIgnored();
      cursor = failed_index + 1u;
      continue;
    }

    if (failed_index != 0u) {
      reactor.changes.erase(reactor.changes.begin(),
                            reactor.changes.begin() +
                                static_cast<std::ptrdiff_t>(failed_index));
    }
    return ProjectApplyFailure(result, failed);
  }

  reactor.changes.clear();
  return ReactorApplyResult::success();
}

bool ReactorChangeQueueAcknowledgeInvalid(
    ReactorRuntime &reactor,
    const ReactorInvalidChangeToken token) noexcept {
  if (!token.valid() || reactor.changes.empty()) {
    return false;
  }
  const ReactorRegistrationChange &front = reactor.changes.front();
  if (!StrictChangeForHandle(front, token.handle()) ||
      front.fd_generation() != token.fd_generation()) {
    return false;
  }
  const std::size_t before = reactor.changes.size();
  reactor.changes.erase(
      std::remove_if(
          reactor.changes.begin(), reactor.changes.end(),
          [handle = token.handle()](
              const ReactorRegistrationChange &change) noexcept {
            return StrictChangeForHandle(change, handle);
          }),
      reactor.changes.end());
  return reactor.changes.size() < before;
}

} // namespace rund::node
