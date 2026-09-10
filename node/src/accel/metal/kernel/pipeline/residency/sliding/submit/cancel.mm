#include "internal.hpp"

namespace rund::node::accel::detail::metal_residency_sliding::submit {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK) &&                 \
    defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0

rund::AccelCheck Cancel(Context &context,
                        std::unique_lock<std::mutex> &terminal_gate,
                        const rund::AccelCheck failure) noexcept {
  terminal_gate.unlock();
  submission::Cancel(context.claim);
  return failure;
}

#endif

} // namespace rund::node::accel::detail::metal_residency_sliding::submit
