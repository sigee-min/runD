#include "local.hpp"
#include "../../../reactor/diagnostics.hpp"

#if !defined(__APPLE__) && !defined(__FreeBSD__) && !defined(__linux__)

#include <cerrno>
namespace rund::node {

ReactorPlatformPollResult PollReactorPlatform(
    ReactorPlatform& handle, const int timeout_ms,
    const std::size_t max_events) noexcept {
  RecordReactorPlatformPoll();
  ReactorPlatformPollResult result{};
  ReactorPlatformState& state = PortableReactorState(handle);
  if (!state.opened || state.registrations.empty() || max_events == 0u) {
    return result;
  }
  state.ready.clear();
  result.ready = &state.ready;
  state.events.clear();
  try {
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
    result.ok = false;
    result.platform_error = ENOMEM;
    return result;
  }
  int native = 0;
  do {
    errno = 0;
    native = ::poll(state.events.data(), state.events.size(), timeout_ms);
  } while (native < 0 && errno == EINTR);
  if (native < 0) {
    result.ok = false;
    result.invalid = errno == EBADF || errno == EINVAL;
    result.platform_error = errno;
    return result;
  }
  try {
    state.ready.reserve(static_cast<std::size_t>(native));
    for (std::size_t index = 0u; index < state.events.size(); ++index) {
      ReactorPlatformReady ready{};
      if (PollReadyEvent(state.events[index], &ready)) {
        state.ready.push_back(ready);
      }
    }
  } catch (...) {
    result.ok = false;
    result.platform_error = ENOMEM;
    state.ready.clear();
  }
  return result;
}

}  // namespace rund::node

#endif
