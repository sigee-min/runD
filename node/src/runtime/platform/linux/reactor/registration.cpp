#include "local.hpp"
#include "../../../reactor/diagnostics.hpp"

#if defined(__linux__)

#include <algorithm>
#include <cerrno>

namespace rund::node {

bool EpollReserveInterestStorage(ReactorPlatform& platform,
                                 const ReactorHandle handle) noexcept {
  for (const ReactorPlatformRegistration& registration :
       LinuxReactorState(platform).registrations) {
    if (registration.handle == handle) return true;
  }
  try {
    LinuxReactorState(platform).registrations.reserve(
        LinuxReactorState(platform).registrations.size() + 1u);
  } catch (...) {
    return false;
  }
  return true;
}

bool EpollRememberInterest(ReactorPlatform& platform,
                           const ReactorHandle handle,
                           const ReactorInterest interest) noexcept {
  for (ReactorPlatformRegistration& registration :
       LinuxReactorState(platform).registrations) {
    if (registration.handle == handle) {
      registration.interest = interest;
      return true;
    }
  }
  try {
    LinuxReactorState(platform).registrations.push_back(
        ReactorPlatformRegistration{.handle = handle, .interest = interest});
  } catch (...) {
    return false;
  }
  return true;
}

void EpollForgetInterest(ReactorPlatform& platform,
                         const ReactorHandle handle) noexcept {
  const auto found =
      std::find_if(LinuxReactorState(platform).registrations.begin(),
                   LinuxReactorState(platform).registrations.end(),
                   [handle](const ReactorPlatformRegistration& registration) {
                     return registration.handle == handle;
                   });
  if (found != LinuxReactorState(platform).registrations.end()) {
    LinuxReactorState(platform).registrations.erase(found);
  }
}

ReactorPlatformOpResult EpollCtl(ReactorPlatform& platform, const int op,
                                 const ReactorHandle handle,
                                 const ReactorInterest interest) noexcept {
  epoll_event event{};
  event.events = EpollEventsForInterest(interest);
  event.data.u64 = handle;
  errno = 0;
  if (::epoll_ctl(LinuxReactorState(platform).native, op, PosixFd(handle),
                  &event) == 0) {
    return ReactorPlatformOpResult::success();
  }
  const int saved_errno = errno;
  return saved_errno == EBADF || saved_errno == EINVAL
             ? ReactorPlatformOpResult::invalid(saved_errno)
             : ReactorPlatformOpResult::failed(saved_errno);
}

ReactorPlatformOpResult AddReactorPlatformInterest(
    ReactorPlatform& platform, const ReactorHandle handle,
    const ReactorInterest interest) noexcept {
  if (!EpollReserveInterestStorage(platform, handle)) {
    return ReactorPlatformOpResult::failed(ENOMEM);
  }
  ReactorPlatformOpResult added =
      EpollCtl(platform, EPOLL_CTL_ADD, handle, interest);
  if (added.disposition() == ReactorPlatformOpDisposition::Failed &&
      added.platform_error() == EEXIST) {
    added = EpollCtl(platform, EPOLL_CTL_MOD, handle, interest);
  }
  if (added.disposition() == ReactorPlatformOpDisposition::Success) {
    if (!EpollRememberInterest(platform, handle, interest)) {
      return ReactorPlatformOpResult::failed(ENOMEM);
    }
    RecordReactorPlatformAdd();
  }
  return added;
}

ReactorPlatformOpResult ModifyReactorPlatformInterest(
    ReactorPlatform& platform, const ReactorHandle handle,
    const ReactorInterest interest) noexcept {
  RecordReactorPlatformModify();
  if (!EpollReserveInterestStorage(platform, handle)) {
    return ReactorPlatformOpResult::failed(ENOMEM);
  }
  ReactorPlatformOpResult modified =
      EpollCtl(platform, EPOLL_CTL_MOD, handle, interest);
  if (modified.disposition() == ReactorPlatformOpDisposition::Failed &&
      modified.platform_error() == ENOENT) {
    modified = EpollCtl(platform, EPOLL_CTL_ADD, handle, interest);
  }
  if (modified.disposition() == ReactorPlatformOpDisposition::Success &&
      !EpollRememberInterest(platform, handle, interest)) {
    return ReactorPlatformOpResult::failed(ENOMEM);
  }
  return modified;
}

ReactorPlatformOpResult RemoveReactorPlatformInterest(
    ReactorPlatform& platform, const ReactorHandle handle) noexcept {
  errno = 0;
  if (::epoll_ctl(LinuxReactorState(platform).native, EPOLL_CTL_DEL,
                  PosixFd(handle), nullptr) == 0 ||
      errno == ENOENT) {
    EpollForgetInterest(platform, handle);
    RecordReactorPlatformRemove();
    return ReactorPlatformOpResult::success();
  }
  const int saved_errno = errno;
  if (saved_errno == EBADF || saved_errno == EINVAL) {
    EpollForgetInterest(platform, handle);
    RecordReactorPlatformRemove();
    return ReactorPlatformOpResult::invalid(saved_errno);
  }
  return ReactorPlatformOpResult::failed(saved_errno);
}

ReactorPlatformBatchResult ApplyReactorPlatformChanges(
    ReactorPlatform& platform, const ReactorRegistrationChange* const changes,
    const std::size_t count) noexcept {
  if (changes == nullptr || count == 0u) {
    return ReactorPlatformBatchResult::success();
  }
  for (std::size_t index = 0u; index < count; ++index) {
    ReactorPlatformOpResult result = ReactorPlatformOpResult::success();
    const ReactorRegistrationChange& change = changes[index];
    switch (change.kind) {
      case ReactorRegistrationChange::Kind::Add:
        result = AddReactorPlatformInterest(platform, change.handle,
                                            change.interest);
        break;
      case ReactorRegistrationChange::Kind::Modify:
        result = ModifyReactorPlatformInterest(platform, change.handle,
                                               change.interest);
        break;
      case ReactorRegistrationChange::Kind::Remove:
        result = RemoveReactorPlatformInterest(platform, change.handle);
        break;
    }
    switch (result.disposition()) {
      case ReactorPlatformOpDisposition::Invalid:
        return ReactorPlatformBatchResult::invalid(result.platform_error(),
                                                   index);
      case ReactorPlatformOpDisposition::Failed:
        return ReactorPlatformBatchResult::failed(result.platform_error(),
                                                  index);
      case ReactorPlatformOpDisposition::BackendUnavailable:
        return ReactorPlatformBatchResult::backend_unavailable();
      case ReactorPlatformOpDisposition::Success:
        break;
    }
  }
  return ReactorPlatformBatchResult::success();
}

}  // namespace rund::node

#endif
