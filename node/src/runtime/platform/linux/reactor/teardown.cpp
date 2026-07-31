#include "local.hpp"
#include "../../../reactor/diagnostics.hpp"

#if defined(__linux__)

namespace rund::node {

void CloseReactorPlatform(ReactorPlatform& platform) noexcept {
  if (!platform.state || !LinuxReactorState(platform).opened) {
    return;
  }
  ReactorPlatformState& state = LinuxReactorState(platform);
  (void)NativeClose(state.native);
  state.native = -1;
  state.opened = false;
  state.registrations.clear();
  RecordReactorPlatformClose();
}

}  // namespace rund::node

#endif
