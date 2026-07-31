#include "local.hpp"

#include "../../../reactor/readiness/mask.hpp"

#if defined(__linux__)

namespace rund::node {

std::uint32_t EpollEventsForInterest(const ReactorInterest interest) noexcept {
  std::uint32_t events = EPOLLERR | EPOLLHUP;
  if (HasReactorInterest(interest, ReactorInterest::Read)) events |= EPOLLIN;
  if (HasReactorInterest(interest, ReactorInterest::Write)) events |= EPOLLOUT;
  return events;
}

ReactorEvent EpollReadyEvents(const std::uint32_t native,
                              const ReactorInterest interest) noexcept {
  ReactorEvent events = ReactorEvent::None;
  if ((native & EPOLLIN) != 0 &&
      HasReactorInterest(interest, ReactorInterest::Read)) {
    events |= ReactorEvent::Read;
  }
  if ((native & EPOLLOUT) != 0 &&
      HasReactorInterest(interest, ReactorInterest::Write)) {
    events |= ReactorEvent::Write;
  }
  if ((native & EPOLLERR) != 0) events |= ReactorEvent::Error;
  if ((native & EPOLLHUP) != 0) events |= ReactorEvent::Hangup;
  return events;
}

ReactorPlatformReady EpollReadyEvent(const ReactorPlatform& platform,
                                     const epoll_event& event) noexcept {
  const ReactorHandle handle = static_cast<ReactorHandle>(event.data.u64);
  const ReactorInterest interest = EpollInterestForHandle(platform, handle);
  return ReactorPlatformReady{
      .handle = handle,
      .events = EpollReadyEvents(event.events, interest),
      .invalid = false,
  };
}

ReactorInterest EpollInterestForHandle(const ReactorPlatform& platform,
                                       const ReactorHandle handle) noexcept {
  for (const ReactorPlatformRegistration& registration :
       LinuxReactorState(platform).registrations) {
    if (registration.handle == handle) return registration.interest;
  }
  return ReactorInterest::None;
}

bool EpollFindInvalidRegistration(const ReactorPlatform& platform,
                                  ReactorPlatformReady* const out) noexcept {
  for (const ReactorPlatformRegistration& registration :
       LinuxReactorState(platform).registrations) {
    if (!NativeFdValid(PosixFd(registration.handle))) {
      if (out != nullptr) {
        *out = ReactorPlatformReady{
            .handle = registration.handle,
            .events = ReactorEventsForInterest(registration.interest),
            .invalid = true,
        };
      }
      return true;
    }
  }
  return false;
}

}  // namespace rund::node

#endif
