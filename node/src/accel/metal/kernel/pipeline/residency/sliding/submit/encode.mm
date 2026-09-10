#include "internal.hpp"

#include "../../../../../../clock.hpp"

namespace rund::node::accel::detail::metal_residency_sliding::submit {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK) &&                 \
    defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0

rund::AccelCheck Encode(Context &context) noexcept {
  context.submit_begin = MonotonicNanoseconds();
  return EncodeMetalResidencySlidingCommand(
      context.sequence, context.authenticated_locals,
      context.native.dispatch_count, context.native.control_count,
      context.native.reset_count, context.native.reset_bytes);
}

#endif

} // namespace rund::node::accel::detail::metal_residency_sliding::submit
