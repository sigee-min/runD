#include "internal.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
MetalResidencyCandidate::~MetalResidencyCandidate() {
  if (submission.watchdog != nil) {
    dispatch_source_cancel(submission.watchdog);
  }
  for (MetalResidencyWindowCommand &command : window.commands) {
    if (command.watchdog != nil) {
      dispatch_source_cancel(command.watchdog);
    }
  }
}
#endif

MetalSequence::~MetalSequence() {
#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  if (@available(macOS 26.0, iOS 26.0, *)) {
    MetalResidencySubmission &submission = residency_submission;
    if (submission.watchdog != nil) {
      dispatch_source_cancel(submission.watchdog);
    }
    MetalResidencySlidingGate &sliding = residency_sliding;
    sliding.active_owner.reset();
    sliding.quarantine_owner.reset();
    sliding.descriptor = nil;
    sliding.pipeline = nil;
    sliding.command = nil;
    sliding.ready = nil;
    MetalResidencyWindow &window = residency_window;
    for (MetalResidencyWindowCommand &command : window.commands) {
      if (command.watchdog != nil) {
        dispatch_source_cancel(command.watchdog);
      }
      if (command.command != nil && command.allocator != nil &&
          (window.phase == MetalResidencyWindowPhase::Cold ||
           window.phase == MetalResidencyWindowPhase::Ready)) {
        [command.command beginCommandBufferWithAllocator:command.allocator];
        [command.command endCommandBuffer];
        [command.allocator reset];
      }
      command.command = nil;
      command.allocator = nil;
      command.watchdog = nil;
    }
    window.ready = nil;
    window.done = nil;
    window.listener = nil;
    window.service = nil;
    if (submission.command != nil && submission.allocator != nil &&
        (submission.phase == MetalResidencySubmissionPhase::Cold ||
         submission.phase == MetalResidencySubmissionPhase::Ready)) {
      [submission.command beginCommandBufferWithAllocator:submission.allocator];
      [submission.command endCommandBuffer];
      [submission.allocator reset];
    }
    [submission.resources endResidency];
    submission.command = nil;
    submission.resources = nil;
    submission.event = nil;
    submission.listener = nil;
    submission.watchdog = nil;
    submission.allocator = nil;
  }
#endif
}

#endif

} // namespace rund::node::accel::detail
