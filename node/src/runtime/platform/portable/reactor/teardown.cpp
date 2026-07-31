#include "local.hpp"
#include "../../../reactor/diagnostics.hpp"

#if !defined(__APPLE__) && !defined(__FreeBSD__) && !defined(__linux__)

namespace rund::node {

void CloseReactorPlatform(ReactorPlatform& platform) noexcept {
  if (!platform.state || !PortableReactorState(platform).opened) {
    return;
  }
  ReactorPlatformState& state = PortableReactorState(platform);
  state.opened = false;
  state.registrations.clear();
  RecordReactorPlatformClose();
}

}  // namespace rund::node

#endif
