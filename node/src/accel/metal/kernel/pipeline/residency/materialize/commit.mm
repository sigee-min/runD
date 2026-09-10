#include "internal.hpp"

#include <rund/counter.hpp>

#include <cassert>
#include <memory>
#include <utility>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

void CommitMetalPipelineResidency(const std::shared_ptr<void> &prepared,
                                  std::shared_ptr<void> owner) noexcept {
#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  if (@available(macOS 26.0, iOS 26.0, *)) {
    auto *const sequence = static_cast<MetalSequence *>(prepared.get());
    auto candidate =
        std::static_pointer_cast<MetalResidencyCandidate>(std::move(owner));
    assert(sequence != nullptr && candidate != nullptr &&
           candidate->sequence == sequence &&
           sequence->memory_meter != nullptr &&
           !sequence->residency_submission.supported());
    [candidate->submission.resources requestResidency];
    candidate->submission.phase = MetalResidencySubmissionPhase::Ready;
    candidate->window.phase = MetalResidencyWindowPhase::Ready;
    MetalResidencySubmission &target = sequence->residency_submission;
    MetalResidencySubmission &source = candidate->submission;
    target.allocator = source.allocator;
    target.command = source.command;
    target.resources = source.resources;
    target.event = source.event;
    target.listener = source.listener;
    target.watchdog = source.watchdog;
    target.owner = source.owner;
    target.phase = source.phase;
    target.event_value = source.event_value;
    target.submit_begin_ns = source.submit_begin_ns;
    target.dispatch_count = source.dispatch_count;
    target.control_count = source.control_count;
    target.reset_count = source.reset_count;
    target.reset_bytes = source.reset_bytes;
    target.native_submit_count = source.native_submit_count;
    target.command_inflight_peak = source.command_inflight_peak;
    target.force_device_lost = source.force_device_lost;
    target.force_terminal_loss = source.force_terminal_loss;
    source.allocator = nil;
    source.command = nil;
    source.resources = nil;
    source.event = nil;
    source.listener = nil;
    source.watchdog = nil;
    source.owner.reset();
    source.phase = MetalResidencySubmissionPhase::Unsupported;
    MetalResidencySlidingGate &sliding_target = sequence->residency_sliding;
    MetalResidencySlidingGate &sliding_source = candidate->sliding;
    sliding_target.descriptor = sliding_source.descriptor;
    sliding_target.pipeline = sliding_source.pipeline;
    sliding_target.command = sliding_source.command;
    sliding_target.ready = sliding_source.ready;
    sliding_target.next_ready_value = sliding_source.next_ready_value;
    sliding_target.retained_bytes = sliding_source.retained_bytes;
    sliding_target.ready_for_submit = sliding_source.ready_for_submit;
    sliding_source.descriptor = nil;
    sliding_source.pipeline = nil;
    sliding_source.command = nil;
    sliding_source.ready = nil;
    sliding_source.retained_bytes = 0u;
    sliding_source.ready_for_submit = false;
    MetalResidencyWindow &window_target = sequence->residency_window;
    MetalResidencyWindow &window_source = candidate->window;
    window_target.ready = window_source.ready;
    window_target.done = window_source.done;
    window_target.listener = window_source.listener;
    window_target.service = window_source.service;
    window_target.owner = window_source.owner;
    window_target.phase = window_source.phase;
    window_target.event_value = window_source.event_value;
    for (std::size_t slot = 0u; slot < window_target.commands.size(); ++slot) {
      MetalResidencyWindowCommand &to = window_target.commands[slot];
      MetalResidencyWindowCommand &from = window_source.commands[slot];
      to.allocator = from.allocator;
      to.command = from.command;
      to.watchdog = from.watchdog;
      from.allocator = nil;
      from.command = nil;
      from.watchdog = nil;
    }
    window_source.ready = nil;
    window_source.done = nil;
    window_source.listener = nil;
    window_source.service = nil;
    window_source.owner.reset();
    window_source.phase = MetalResidencyWindowPhase::Unsupported;
    sequence->memory_meter->add_device(
        PreparedMemory{.current = candidate->retained_bytes,
                       .peak = candidate->retained_bytes,
                       .cumulative = candidate->retained_bytes,
                       .budget = candidate->retained_bytes});
    sequence->retained_bytes = ::rund::detail::counter::SaturatingAdd(
        sequence->retained_bytes, candidate->retained_bytes);
  }
#else
  static_cast<void>(prepared);
  static_cast<void>(owner);
#endif
}

#endif

} // namespace rund::node::accel::detail
