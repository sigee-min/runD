#include "local.hpp"

#include "../../../reactor/readiness/mask.hpp"

#if defined(__APPLE__) || defined(__FreeBSD__)

namespace rund::node {

ReactorEvent KqueueReadyEvents(const struct kevent& event,
                               const ReactorInterest interest) noexcept {
  ReactorEvent events = ReactorEvent::None;
  if (event.filter == EVFILT_READ &&
      HasReactorInterest(interest, ReactorInterest::Read)) {
    events |= ReactorEvent::Read;
  }
  if (event.filter == EVFILT_WRITE &&
      HasReactorInterest(interest, ReactorInterest::Write)) {
    events |= ReactorEvent::Write;
  }
  if ((event.flags & EV_EOF) != 0) {
    events |= ReactorEvent::Hangup;
  }
  if ((event.flags & EV_ERROR) != 0 && event.data != 0) {
    events |= ReactorEvent::Error;
  }
  return events;
}

ReactorInterest KqueueInterestForHandle(const ReactorPlatform& platform,
                                        const ReactorHandle handle) noexcept {
  for (const ReactorPlatformRegistration& registration :
       MacReactorState(platform).registrations) {
    if (registration.handle == handle) {
      return registration.interest;
    }
  }
  return ReactorInterest::None;
}

bool KqueueFindInvalidRegistration(const ReactorPlatform& platform,
                                   ReactorPlatformReady* const out) noexcept {
  for (const ReactorPlatformRegistration& registration :
       MacReactorState(platform).registrations) {
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
