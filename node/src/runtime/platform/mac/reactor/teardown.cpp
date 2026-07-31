#include "local.hpp"
#include "../../../reactor/diagnostics.hpp"

#if defined(__APPLE__) || defined(__FreeBSD__)

namespace rund::node {

void CloseReactorPlatform(ReactorPlatform& platform) noexcept {
  if (!platform.state || !MacReactorState(platform).opened) {
    return;
  }
  ReactorPlatformState& state = MacReactorState(platform);
  (void)NativeClose(state.native);
  state.native = -1;
  state.opened = false;
  state.registrations.clear();
  RecordReactorPlatformClose();
}

}  // namespace rund::node

#endif
