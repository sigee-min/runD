#include "internal.hpp"

namespace rund::node::accel::detail::metal_residency_sliding::submit {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK) &&                 \
    defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0

Context Capture(const std::shared_ptr<void> &prepared,
                const BackendResidencySlidingDescriptor &descriptor,
                const std::span<const std::uint32_t> locals,
                MetalSequence &sequence) noexcept {
  return Context{
      .prepared = prepared,
      .descriptor = descriptor,
      .locals = locals,
      .sequence = sequence,
      .claim = sequence.submission,
      .adapter = *sequence.adapter,
      .native = sequence.residency_submission,
      .gate = sequence.residency_sliding,
      .queue = MetalResidencyScheduleQueue(sequence),
      .guard = [sequence.guard_zero contents],
      .destination = [sequence.residency_sliding.descriptor contents],
  };
}

#endif

} // namespace rund::node::accel::detail::metal_residency_sliding::submit
