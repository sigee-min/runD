#include "local.hpp"
#include "../../../reactor/diagnostics.hpp"

#if defined(__linux__)

#include <cerrno>
#include <new>

namespace rund::node {

void ReactorPlatformStateDelete::operator()(
    ReactorPlatformState* const state) const noexcept {
  if (state == nullptr) {
    return;
  }
  if (state->opened) {
    (void)NativeClose(state->native);
    state->opened = false;
    RecordReactorPlatformClose();
  }
  delete state;
}

ReactorPlatformOpResult PrepareReactorPlatform(
    ReactorPlatform& platform, const std::size_t capacity) noexcept {
  if (!platform.state) {
    platform.state.reset(new (std::nothrow) ReactorPlatformState{});
    if (!platform.state) {
      return ReactorPlatformOpResult{.ok = false, .platform_error = ENOMEM};
    }
  }
  ReactorPlatformState& state = LinuxReactorState(platform);
  try {
    state.registrations.reserve(capacity);
    state.events.resize(capacity);
    state.probe.events.reserve(capacity);
  } catch (...) {
    return ReactorPlatformOpResult{.ok = false, .platform_error = ENOMEM};
  }
  return {};
}

ReactorPlatformOpResult OpenReactorPlatform(
    ReactorPlatform& platform) noexcept {
  const ReactorPlatformOpResult prepared = PrepareReactorPlatform(platform, 0u);
  if (!prepared.ok) {
    return prepared;
  }
  ReactorPlatformState& state = LinuxReactorState(platform);
  if (state.opened) {
    return {};
  }
  errno = 0;
  const int native = ::epoll_create1(EPOLL_CLOEXEC);
  if (native < 0) {
    return ReactorPlatformOpResult{.ok = false, .platform_error = errno};
  }
  state.native = native;
  state.opened = true;
  RecordReactorPlatformOpen();
  return {};
}

}  // namespace rund::node

#endif
