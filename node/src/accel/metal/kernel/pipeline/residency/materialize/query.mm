#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck
QueryMetalPipelineResidency(const std::shared_ptr<void> &prepared,
                            bool &supported) noexcept {
  supported = false;
  auto *const sequence = static_cast<MetalSequence *>(prepared.get());
  if (!ValidMetalSequence(sequence)) {
    return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
  }
#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  if (@available(macOS 26.0, iOS 26.0, *)) {
    supported = sequence->residency_selectable;
  }
#endif
  return rund::AccelCheck{true, "ok"};
}

rund::AccelCheck
MetalPipelineResidencyReady(const std::shared_ptr<void> &prepared,
                            bool &ready) noexcept {
  ready = false;
  auto *const sequence = static_cast<MetalSequence *>(prepared.get());
  if (!ValidMetalSequence(sequence)) {
    return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
  }
#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  if (@available(macOS 26.0, iOS 26.0, *)) {
    ready = sequence->adapter != nullptr &&
            !sequence->adapter->residency_quarantined.load(
                std::memory_order_acquire) &&
            !sequence->adapter->fault_host_read_once.load(
                std::memory_order_relaxed) &&
            !sequence->adapter->fault_host_write_once.load(
                std::memory_order_relaxed) &&
            sequence->residency_selectable && !sequence->direct_aggregate &&
            sequence->state_count == 0u && sequence->guard_zero != nil &&
            [sequence->guard_zero contents] != nullptr &&
            sequence->residency_window.service != nil &&
            sequence->residency_submission.ready() &&
            sequence->residency_window.ready_for_submit();
  }
#endif
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
