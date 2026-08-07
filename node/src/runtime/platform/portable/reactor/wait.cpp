#include "local.hpp"
#include "../../../reactor/diagnostics.hpp"

#if !defined(__APPLE__) && !defined(__FreeBSD__) && !defined(__linux__)

#include <cerrno>
namespace rund::node {

ReactorPlatformPollResult PollReactorPlatform(
    ReactorPlatform& handle, const int timeout_ms,
    const std::size_t max_events,
    std::vector<ReactorPlatformReady>& out) noexcept {
  RecordReactorPlatformPoll();
  out.clear();
  ReactorPlatformState& state = PortableReactorState(handle);
  if (!state.opened || state.registrations.empty() || max_events == 0u) {
    return ReactorPlatformPollResult::success();
  }
  state.events.clear();
  try {
    out.reserve(state.registrations.size());
    state.events.reserve(state.registrations.size());
    for (const ReactorPlatformRegistration& registration :
         state.registrations) {
      state.events.push_back(pollfd{
          .fd = PosixFd(registration.handle),
          .events = PosixInterest(registration.interest),
          .revents = 0,
      });
    }
  } catch (...) {
    return ReactorPlatformPollResult::failed(ENOMEM);
  }
  int native = 0;
  do {
    errno = 0;
    native = ::poll(state.events.data(), state.events.size(), timeout_ms);
  } while (native < 0 && errno == EINTR);
  if (native < 0) {
    return errno == EBADF || errno == EINVAL
               ? ReactorPlatformPollResult::invalid(errno)
               : ReactorPlatformPollResult::failed(errno);
  }
  try {
    for (std::size_t index = 0u; index < state.events.size(); ++index) {
      ReactorPlatformReady ready{};
      if (PollReadyEvent(state.events[index], &ready)) {
        out.push_back(ready);
      }
    }
  } catch (...) {
    out.clear();
    return ReactorPlatformPollResult::failed(ENOMEM);
  }
  return ReactorPlatformPollResult::success();
}

}  // namespace rund::node

#endif
