#include "queue.hpp"

#include <algorithm>
#include <cstddef>

#include "../../../../reactor/diagnostics.hpp"
#include "../../../../reactor/platform.hpp"
#include "../stats.hpp"

namespace rund::node {
namespace {

[[nodiscard]] bool
IgnorableInvalidRemove(const ReactorRegistrationChange &change,
                       const ReactorPlatformBatchResult &result) noexcept {
  return change.best_effort &&
         change.kind == ReactorRegistrationChange::Kind::Remove &&
         result.invalid;
}

void RecordApplied(::rund::detail::task::StatStorage &stats,
                   const std::size_t count) noexcept {
  for (std::size_t index = 0u; index < count; ++index) {
    RecordReactorRegistrationChangeApplied(stats);
  }
}

[[nodiscard]] ReactorApplyResult
ProjectApplyFailure(const ReactorPlatformBatchResult &result,
                    const ReactorHandle failed_handle) noexcept {
  if (result.invalid) {
    return ReactorApplyResult::invalid(failed_handle);
  }
  if (result.unavailable) {
    return ReactorApplyResult::backend_unavailable();
  }
  return ReactorApplyResult::failed();
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
    if (result.ok) {
      RecordApplied(stats, remaining);
      reactor.changes.clear();
      return ReactorApplyResult::success();
    }

    const std::size_t failed_offset =
        std::min<std::size_t>(result.failed_index, remaining - 1u);
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
    return ProjectApplyFailure(result, failed.handle);
  }

  reactor.changes.clear();
  return ReactorApplyResult::success();
}

} // namespace rund::node
