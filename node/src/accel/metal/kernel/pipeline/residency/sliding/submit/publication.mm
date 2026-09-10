#include "internal.hpp"

#include <atomic>
#include <cstring>

namespace rund::node::accel::detail::metal_residency_sliding::submit {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK) &&                 \
    defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0

void PublishDescriptor(Context &context) noexcept {
  *static_cast<std::uint32_t *>(context.guard) = 1u;
  std::memcpy(context.destination, &context.storage.payload,
              sizeof(context.storage.payload));
  std::atomic_thread_fence(std::memory_order_release);
  context.ready_value = context.gate.next_ready_value++;
  context.gate.ready.signaledValue = context.ready_value;
}

#endif

} // namespace rund::node::accel::detail::metal_residency_sliding::submit
