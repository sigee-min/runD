#include "local.hpp"
#include "../../../reactor/diagnostics.hpp"

#if !defined(__APPLE__) && !defined(__FreeBSD__) && !defined(__linux__)

#include <cerrno>
#include <new>

namespace rund::node {

void ReactorPlatformStateDelete::operator()(
    ReactorPlatformState* const state) const noexcept {
  if (state != nullptr && state->opened) {
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
      return ReactorPlatformOpResult::failed(ENOMEM);
    }
  }
  ReactorPlatformState& state = PortableReactorState(platform);
  try {
    state.registrations.reserve(capacity);
    state.events.reserve(capacity);
    state.probe.events.reserve(capacity);
  } catch (...) {
    return ReactorPlatformOpResult::failed(ENOMEM);
  }
  return ReactorPlatformOpResult::success();
}

ReactorPlatformOpResult OpenReactorPlatform(
    ReactorPlatform& platform) noexcept {
  const ReactorPlatformOpResult prepared = PrepareReactorPlatform(platform, 0u);
  if (prepared.disposition() != ReactorPlatformOpDisposition::Success) {
    return prepared;
  }
  ReactorPlatformState& state = PortableReactorState(platform);
  if (state.opened) {
    return ReactorPlatformOpResult::success();
  }
  state.opened = true;
  RecordReactorPlatformOpen();
  return ReactorPlatformOpResult::success();
}

}  // namespace rund::node

#endif
