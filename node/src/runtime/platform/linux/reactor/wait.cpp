#include "local.hpp"
#include "../../../reactor/diagnostics.hpp"

#if defined(__linux__)

#include <cerrno>
namespace rund::node {

ReactorPlatformPollResult PollReactorPlatform(
    ReactorPlatform& handle, const int timeout_ms,
    const std::size_t max_events) noexcept {
  RecordReactorPlatformPoll();
  ReactorPlatformPollResult result{};
  ReactorPlatformState& state = LinuxReactorState(handle);
  if (!state.opened || max_events == 0u) {
    return result;
  }
  state.ready.clear();
  result.ready = &state.ready;
  if (timeout_ms < 0) {
    ReactorPlatformReady invalid{};
    if (EpollFindInvalidRegistration(handle, &invalid)) {
      state.ready.push_back(invalid);
      return result;
    }
  }
  try {
    state.events.resize(max_events);
  } catch (...) {
    result.ok = false;
    result.platform_error = ENOMEM;
    return result;
  }
  int native = 0;
  do {
    errno = 0;
    native = ::epoll_wait(state.native, state.events.data(),
                          static_cast<int>(state.events.size()), timeout_ms);
  } while (native < 0 && errno == EINTR);
  const int saved_errno = errno;
  if (native < 0) {
    result.ok = false;
    result.invalid = saved_errno == EBADF || saved_errno == EINVAL;
    result.platform_error = saved_errno;
    return result;
  }
  try {
    state.ready.reserve(static_cast<std::size_t>(native));
    for (int index = 0; index < native; ++index) {
      const epoll_event& event = state.events[static_cast<std::size_t>(index)];
      state.ready.push_back(EpollReadyEvent(handle, event));
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
