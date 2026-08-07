#include "local.hpp"
#include "../../../reactor/diagnostics.hpp"

#if defined(__APPLE__) || defined(__FreeBSD__)

#include <cerrno>
#include <cstdint>
#include <limits>

namespace rund::node {

timespec KqueueTimeoutSpec(const int timeout_ms) noexcept {
  if (timeout_ms < 0) {
    return timespec{.tv_sec = 0, .tv_nsec = 0};
  }
  return timespec{.tv_sec = timeout_ms / 1000,
                  .tv_nsec = (timeout_ms % 1000) * 1000000};
}

ReactorPlatformPollResult PollReactorPlatform(
    ReactorPlatform& handle, const int timeout_ms,
    const std::size_t max_events,
    std::vector<ReactorPlatformReady>& out) noexcept {
  RecordReactorPlatformPoll();
  out.clear();
  if (!MacReactorState(handle).opened || max_events == 0u) {
    return ReactorPlatformPollResult::success();
  }
  if (max_events > std::numeric_limits<std::size_t>::max() / 2u) {
    return ReactorPlatformPollResult::failed(ENOMEM);
  }
  const std::size_t event_capacity = max_events * 2u;
  try {
    out.reserve(event_capacity);
  } catch (...) {
    return ReactorPlatformPollResult::failed(ENOMEM);
  }
  if (timeout_ms < 0) {
    ReactorPlatformReady invalid{};
    if (KqueueFindInvalidRegistration(handle, &invalid)) {
      out.push_back(invalid);
      return ReactorPlatformPollResult::success();
    }
  }
  std::vector<struct kevent>& events = MacReactorState(handle).events;
  try {
    events.resize(event_capacity);
  } catch (...) {
    return ReactorPlatformPollResult::failed(ENOMEM);
  }
  timespec timeout = KqueueTimeoutSpec(timeout_ms);
  timespec* const timeout_ptr = timeout_ms < 0 ? nullptr : &timeout;
  int native = 0;
  do {
    errno = 0;
    native = ::kevent(MacReactorState(handle).native, nullptr, 0, events.data(),
                      static_cast<int>(events.size()), timeout_ptr);
  } while (native < 0 && errno == EINTR);
  const int saved_errno = errno;
  if (native < 0) {
    return saved_errno == EBADF || saved_errno == EINVAL
               ? ReactorPlatformPollResult::invalid(saved_errno)
               : ReactorPlatformPollResult::failed(saved_errno);
  }
  try {
    for (int index = 0; index < native; ++index) {
      const struct kevent& event = events[static_cast<std::size_t>(index)];
      const ReactorHandle native_handle =
          static_cast<ReactorHandle>(reinterpret_cast<uintptr_t>(event.udata));
      const ReactorInterest interest =
          KqueueInterestForHandle(handle, native_handle);
      const bool invalid = ((event.flags & EV_ERROR) != 0 &&
                            (event.data == EBADF || event.data == EINVAL));
      out.push_back(ReactorPlatformReady{
          .handle = native_handle,
          .events = KqueueReadyEvents(event, interest),
          .invalid = invalid,
      });
    }
  } catch (...) {
    out.clear();
    return ReactorPlatformPollResult::failed(ENOMEM);
  }
  return ReactorPlatformPollResult::success();
}

}  // namespace rund::node

#endif
