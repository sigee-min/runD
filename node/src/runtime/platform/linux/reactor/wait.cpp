#include "local.hpp"
#include "../../../reactor/diagnostics.hpp"

#if defined(__linux__)

#include <cerrno>
namespace rund::node {

ReactorPlatformPollResult PollReactorPlatform(
    ReactorPlatform& handle, const int timeout_ms,
    const std::size_t max_events,
    std::vector<ReactorPlatformReady>& out) noexcept {
  RecordReactorPlatformPoll();
  out.clear();
  ReactorPlatformState& state = LinuxReactorState(handle);
  if (!state.opened || max_events == 0u) {
    return ReactorPlatformPollResult::success();
  }
  try {
    out.reserve(max_events);
  } catch (...) {
    return ReactorPlatformPollResult::failed(ENOMEM);
  }
  if (timeout_ms < 0) {
    ReactorPlatformReady invalid{};
    if (EpollFindInvalidRegistration(handle, &invalid)) {
      out.push_back(invalid);
      return ReactorPlatformPollResult::success();
    }
  }
  try {
    state.events.resize(max_events);
  } catch (...) {
    return ReactorPlatformPollResult::failed(ENOMEM);
  }
  int native = 0;
  do {
    errno = 0;
    native = ::epoll_wait(state.native, state.events.data(),
                          static_cast<int>(state.events.size()), timeout_ms);
  } while (native < 0 && errno == EINTR);
  const int saved_errno = errno;
  if (native < 0) {
    return saved_errno == EBADF || saved_errno == EINVAL
               ? ReactorPlatformPollResult::invalid(saved_errno)
               : ReactorPlatformPollResult::failed(saved_errno);
  }
  try {
    for (int index = 0; index < native; ++index) {
      const epoll_event& event = state.events[static_cast<std::size_t>(index)];
      out.push_back(EpollReadyEvent(handle, event));
    }
  } catch (...) {
    out.clear();
    return ReactorPlatformPollResult::failed(ENOMEM);
  }
  return ReactorPlatformPollResult::success();
}

}  // namespace rund::node

#endif
