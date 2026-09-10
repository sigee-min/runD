#include "internal.hpp"

#include <array>

namespace rund::node::accel::detail::metal_residency_sliding::submit {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK) &&                 \
    defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0

void CommitQueue(Context &context) noexcept {
  [context.queue waitForEvent:context.gate.ready value:context.ready_value];
  const id<MTL4CommandBuffer> commands[] = {context.native.command};
  [context.queue commit:commands count:1u];
  // Count the actual accepted queue call at commit, not only after a Known
  // retirement. A watchdog Unknown still consumed exactly one queue call.
  RecordMetalCommandSubmit(context.adapter);
  context.native.command_inflight_peak =
      BeginMetalResidencyCommand(context.adapter);
}

#endif

} // namespace rund::node::accel::detail::metal_residency_sliding::submit
