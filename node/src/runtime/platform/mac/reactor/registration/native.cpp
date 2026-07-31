#include "../local.hpp"

#include "../../../../reactor/readiness/mask.hpp"

#if defined(__APPLE__) || defined(__FreeBSD__)

#include <algorithm>
#include <cerrno>
#include <cstdint>

namespace rund::node {

void KqueuePushFilter(std::vector<struct kevent>& changes,
                      const ReactorHandle handle, const short filter,
                      const std::uint16_t flags) {
  const int fd = PosixFd(handle);
  struct kevent event{};
  EV_SET(&event, static_cast<uintptr_t>(fd), filter, flags, 0, 0,
         reinterpret_cast<void*>(static_cast<intptr_t>(fd)));
  changes.push_back(event);
}

void KqueuePushReceiptFilter(std::vector<struct kevent>& changes,
                             std::vector<std::size_t>& owners,
                             std::vector<bool>& ignore_missing,
                             const std::size_t owner, const bool ignore_enoent,
                             const ReactorHandle handle, const short filter,
                             const std::uint16_t flags) {
  KqueuePushFilter(changes, handle, filter,
                   static_cast<std::uint16_t>(flags | EV_RECEIPT));
  owners.push_back(owner);
  ignore_missing.push_back(ignore_enoent);
}

ReactorPlatformOpResult KqueueSubmit(ReactorPlatform& platform,
                                     struct kevent* const changes,
                                     const int count,
                                     const bool ignore_missing) noexcept {
  if (ignore_missing) {
    auto& native_changes = MacReactorState(platform).changes;
    auto& owners = MacReactorState(platform).owners;
    auto& ignored = MacReactorState(platform).ignore_missing;
    try {
      native_changes.clear();
      owners.clear();
      ignored.clear();
      native_changes.reserve(static_cast<std::size_t>(count));
      owners.reserve(static_cast<std::size_t>(count));
      ignored.reserve(static_cast<std::size_t>(count));
      for (int index = 0; index < count; ++index) {
        struct kevent change = changes[index];
        change.flags = static_cast<std::uint16_t>(change.flags | EV_RECEIPT);
        native_changes.push_back(change);
        owners.push_back(0u);
        ignored.push_back(true);
      }
    } catch (...) {
      return ReactorPlatformOpResult{.ok = false, .platform_error = ENOMEM};
    }
    const ReactorPlatformBatchResult submitted = KqueueSubmitBatch(platform);
    return ReactorPlatformOpResult{
        .ok = submitted.ok,
        .invalid = submitted.invalid,
        .unavailable = submitted.unavailable,
        .platform_error = submitted.platform_error,
    };
  }

  errno = 0;
  const int rc = ::kevent(MacReactorState(platform).native, changes, count,
                          nullptr, 0, nullptr);
  if (rc == 0) {
    return {};
  }
  const int saved_errno = errno;
  return ReactorPlatformOpResult{
      .ok = false,
      .invalid = saved_errno == EBADF || saved_errno == EINVAL,
      .platform_error = saved_errno,
  };
}

ReactorPlatformBatchResult KqueueSubmitBatch(ReactorPlatform& handle) noexcept {
  auto& changes = MacReactorState(handle).changes;
  auto& owners = MacReactorState(handle).owners;
  auto& ignore_missing = MacReactorState(handle).ignore_missing;
  if (changes.empty()) {
    return {};
  }
  std::vector<struct kevent>& receipts = MacReactorState(handle).receipts;
  try {
    receipts.resize(changes.size());
  } catch (...) {
    return ReactorPlatformBatchResult{.ok = false, .platform_error = ENOMEM};
  }
  errno = 0;
  const int rc = ::kevent(MacReactorState(handle).native, changes.data(),
                          static_cast<int>(changes.size()), receipts.data(),
                          static_cast<int>(receipts.size()), nullptr);
  const int saved_errno = errno;
  if (rc < 0) {
    return ReactorPlatformBatchResult{
        .ok = false,
        .invalid = saved_errno == EBADF || saved_errno == EINVAL,
        .platform_error = saved_errno,
        .failed_index = owners.empty() ? 0u : owners.front(),
    };
  }
  const std::size_t receipt_count =
      std::min<std::size_t>(static_cast<std::size_t>(rc), receipts.size());
  for (std::size_t index = 0u; index < receipt_count; ++index) {
    const struct kevent& receipt = receipts[index];
    if ((receipt.flags & EV_ERROR) == 0 || receipt.data == 0) {
      continue;
    }
    const int event_errno = static_cast<int>(receipt.data);
    if (ignore_missing[index] && event_errno == ENOENT) {
      continue;
    }
    return ReactorPlatformBatchResult{
        .ok = false,
        .invalid = event_errno == EBADF || event_errno == EINVAL,
        .platform_error = event_errno,
        .failed_index = owners[index],
    };
  }
  return {};
}

ReactorPlatformOpResult KqueueAddInterest(
    ReactorPlatform& platform, const ReactorHandle handle,
    const ReactorInterest interest) noexcept {
  const int fd = PosixFd(handle);
  struct kevent changes[2]{};
  int count = 0;
  const auto append = [&](const short filter) {
    EV_SET(&changes[static_cast<std::size_t>(count++)],
           static_cast<uintptr_t>(fd), filter, EV_ADD, 0, 0,
           reinterpret_cast<void*>(static_cast<intptr_t>(fd)));
  };
  if (HasReactorInterest(interest, ReactorInterest::Read)) append(EVFILT_READ);
  if (HasReactorInterest(interest, ReactorInterest::Write)) {
    append(EVFILT_WRITE);
  }
  if (count == 0) {
    return {};
  }
  return KqueueSubmit(platform, changes, count, false);
}

ReactorPlatformOpResult KqueueRemoveInterest(
    ReactorPlatform& platform, const ReactorHandle handle,
    const bool ignore_missing) noexcept {
  const int fd = PosixFd(handle);
  struct kevent changes[2]{};
  EV_SET(&changes[0], static_cast<uintptr_t>(fd), EVFILT_READ, EV_DELETE, 0, 0,
         reinterpret_cast<void*>(static_cast<intptr_t>(fd)));
  EV_SET(&changes[1], static_cast<uintptr_t>(fd), EVFILT_WRITE, EV_DELETE, 0, 0,
         reinterpret_cast<void*>(static_cast<intptr_t>(fd)));
  return KqueueSubmit(platform, changes, 2, ignore_missing);
}

}  // namespace rund::node

#endif
