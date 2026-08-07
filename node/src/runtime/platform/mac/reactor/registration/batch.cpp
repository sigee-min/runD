#include "../local.hpp"

#include "../../../../reactor/readiness/mask.hpp"

#if defined(__APPLE__) || defined(__FreeBSD__)

#include <algorithm>
#include <cerrno>

namespace rund::node {
namespace {

[[nodiscard]] ReactorPlatformBatchResult
RebaseBatchFailure(const ReactorPlatformBatchResult result,
                   const std::size_t offset) noexcept {
  switch (result.disposition()) {
  case ReactorPlatformBatchDisposition::Invalid:
    return ReactorPlatformBatchResult::invalid(result.platform_error(),
                                               offset + result.failed_index());
  case ReactorPlatformBatchDisposition::Failed:
    return ReactorPlatformBatchResult::failed(result.platform_error(),
                                              offset + result.failed_index());
  case ReactorPlatformBatchDisposition::BackendUnavailable:
    return ReactorPlatformBatchResult::backend_unavailable();
  case ReactorPlatformBatchDisposition::Success:
    return ReactorPlatformBatchResult::success();
  }
}

} // namespace

ReactorPlatformOpResult KqueueApplyOneChange(
    ReactorPlatform& handle, const ReactorRegistrationChange& change) noexcept {
  switch (change.kind) {
    case ReactorRegistrationChange::Kind::Add:
      return AddReactorPlatformInterest(handle, change.handle, change.interest);
    case ReactorRegistrationChange::Kind::Modify:
      return ModifyReactorPlatformInterest(handle, change.handle,
                                           change.interest);
    case ReactorRegistrationChange::Kind::Remove:
      return RemoveReactorPlatformInterest(handle, change.handle);
  }
  return ReactorPlatformOpResult::success();
}

ReactorPlatformBatchResult ApplyReactorPlatformChanges(
    ReactorPlatform& handle, const ReactorRegistrationChange* const changes,
    const std::size_t count) noexcept {
  if (changes == nullptr || count == 0u) {
    return ReactorPlatformBatchResult::success();
  }

  for (std::size_t index = 0u; index < count; ++index) {
    const ReactorRegistrationChange& change = changes[index];
    const bool isolated_modify =
        change.kind == ReactorRegistrationChange::Kind::Modify;
    const bool isolated_remove =
        change.best_effort &&
        change.kind == ReactorRegistrationChange::Kind::Remove;
    if (isolated_modify || isolated_remove) {
      if (index != 0u) {
        const ReactorPlatformBatchResult prefix =
            ApplyReactorPlatformChanges(handle, changes, index);
        if (prefix.disposition() != ReactorPlatformBatchDisposition::Success) {
          return prefix;
        }
      }
      const ReactorPlatformOpResult one = KqueueApplyOneChange(handle, change);
      switch (one.disposition()) {
      case ReactorPlatformOpDisposition::Invalid:
        return ReactorPlatformBatchResult::invalid(one.platform_error(),
                                                   index);
      case ReactorPlatformOpDisposition::Failed:
        return ReactorPlatformBatchResult::failed(one.platform_error(), index);
      case ReactorPlatformOpDisposition::BackendUnavailable:
        return ReactorPlatformBatchResult::backend_unavailable();
      case ReactorPlatformOpDisposition::Success:
        break;
      }
      if (index + 1u == count) {
        return ReactorPlatformBatchResult::success();
      }
      const ReactorPlatformBatchResult suffix = ApplyReactorPlatformChanges(
          handle, changes + index + 1u, count - index - 1u);
      if (suffix.disposition() != ReactorPlatformBatchDisposition::Success) {
        return RebaseBatchFailure(suffix, index + 1u);
      }
      return ReactorPlatformBatchResult::success();
    }
  }

  if (!KqueueReserveBatchInterestStorage(handle, changes, count)) {
    return ReactorPlatformBatchResult::failed(ENOMEM, 0u);
  }

  auto& native_changes = MacReactorState(handle).changes;
  auto& owners = MacReactorState(handle).owners;
  auto& ignore_missing = MacReactorState(handle).ignore_missing;
  try {
    native_changes.clear();
    owners.clear();
    ignore_missing.clear();
    native_changes.reserve(count * 4u);
    owners.reserve(count * 4u);
    ignore_missing.reserve(count * 4u);
    for (std::size_t index = 0u; index < count; ++index) {
      const ReactorRegistrationChange& change = changes[index];
      switch (change.kind) {
        case ReactorRegistrationChange::Kind::Add:
          if (HasReactorInterest(change.interest, ReactorInterest::Read)) {
            KqueuePushReceiptFilter(native_changes, owners, ignore_missing,
                                    index, false, change.handle, EVFILT_READ,
                                    EV_ADD);
          }
          if (HasReactorInterest(change.interest, ReactorInterest::Write)) {
            KqueuePushReceiptFilter(native_changes, owners, ignore_missing,
                                    index, false, change.handle, EVFILT_WRITE,
                                    EV_ADD);
          }
          break;
        case ReactorRegistrationChange::Kind::Modify:
          KqueuePushReceiptFilter(native_changes, owners, ignore_missing, index,
                                  true, change.handle, EVFILT_READ, EV_DELETE);
          KqueuePushReceiptFilter(native_changes, owners, ignore_missing, index,
                                  true, change.handle, EVFILT_WRITE, EV_DELETE);
          if (HasReactorInterest(change.interest, ReactorInterest::Read)) {
            KqueuePushReceiptFilter(native_changes, owners, ignore_missing,
                                    index, false, change.handle, EVFILT_READ,
                                    EV_ADD);
          }
          if (HasReactorInterest(change.interest, ReactorInterest::Write)) {
            KqueuePushReceiptFilter(native_changes, owners, ignore_missing,
                                    index, false, change.handle, EVFILT_WRITE,
                                    EV_ADD);
          }
          break;
        case ReactorRegistrationChange::Kind::Remove:
          KqueuePushReceiptFilter(native_changes, owners, ignore_missing, index,
                                  change.best_effort, change.handle,
                                  EVFILT_READ, EV_DELETE);
          KqueuePushReceiptFilter(native_changes, owners, ignore_missing, index,
                                  change.best_effort, change.handle,
                                  EVFILT_WRITE, EV_DELETE);
          break;
      }
    }
  } catch (...) {
    return ReactorPlatformBatchResult::failed(ENOMEM, 0u);
  }

  const ReactorPlatformBatchResult submitted = KqueueSubmitBatch(handle);
  if (submitted.disposition() != ReactorPlatformBatchDisposition::Success) {
    KqueueRememberBatchInterest(
        handle, changes,
        std::min<std::size_t>(submitted.failed_index(), count));
    return submitted;
  }
  KqueueRememberBatchInterest(handle, changes, count);
  return ReactorPlatformBatchResult::success();
}

}  // namespace rund::node

#endif
