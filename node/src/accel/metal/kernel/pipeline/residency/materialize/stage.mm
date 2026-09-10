#include "internal.hpp"

#include "../../../../../clock.hpp"
#include "../../../../pipeline/named.hpp"
#include "../../icb.hpp"
#include "../sliding/internal.hpp"

#include <limits>
#include <memory>
#include <new>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
[[nodiscard]] bool
MetalResidencyRetainedBytes(const MetalResidencySubmission &submission,
                            const MetalResidencySlidingGate &sliding,
                            const MetalResidencyWindow &window,
                            std::uint64_t &retained_bytes) noexcept {
  if (submission.allocator == nil || submission.resources == nil) {
    return false;
  }
  const std::uint64_t allocator_bytes =
      static_cast<std::uint64_t>(submission.allocator.allocatedSize);
  const std::uint64_t resource_bytes =
      static_cast<std::uint64_t>(submission.resources.allocatedSize);
  if (allocator_bytes >
      std::numeric_limits<std::uint64_t>::max() - resource_bytes) {
    return false;
  }
  retained_bytes = allocator_bytes + resource_bytes;
  if (sliding.ready_for_submit) {
    if (sliding.retained_bytes == 0u ||
        retained_bytes > std::numeric_limits<std::uint64_t>::max() -
                             sliding.retained_bytes) {
      return false;
    }
    retained_bytes += sliding.retained_bytes;
  }
  for (const MetalResidencyWindowCommand &command : window.commands) {
    if (command.allocator == nil) {
      return false;
    }
    const std::uint64_t bytes =
        static_cast<std::uint64_t>(command.allocator.allocatedSize);
    if (retained_bytes > std::numeric_limits<std::uint64_t>::max() - bytes) {
      return false;
    }
    retained_bytes += bytes;
  }
  return true;
}

#endif

} // namespace

rund::AccelCheck
StageMetalPipelineResidency(const std::shared_ptr<void> &prepared,
                            std::shared_ptr<void> &owner,
                            std::uint64_t &retained_bytes) noexcept {
  owner.reset();
  retained_bytes = 0u;
  auto *const sequence = static_cast<MetalSequence *>(prepared.get());
  if (!ValidMetalSequence(sequence) || !sequence->residency_selectable ||
      (sequence->persistent_spatial_window_selectable &&
       !sequence->spatial_window_proof_valid()) ||
      sequence->residency_submission.supported()) {
    return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
  }
#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  if (@available(macOS 26.0, iOS 26.0, *)) {
    if (sequence->adapter == nullptr || sequence->warm.resources == nullptr ||
        sequence->warm.resource_count == 0u ||
        sequence->warm.chunks == nullptr || sequence->warm.chunk_count == 0u ||
        sequence->pipelines.empty() || sequence->memory_meter == nullptr ||
        sequence->warm.resource_count > std::numeric_limits<NSUInteger>::max() -
                                            sequence->warm.chunk_count ||
        sequence->warm.resource_count + sequence->warm.chunk_count >
            std::numeric_limits<NSUInteger>::max() -
                sequence->pipelines.size() ||
        sequence->warm.resource_count + sequence->warm.chunk_count +
                sequence->pipelines.size() >
            std::numeric_limits<NSUInteger>::max() - 3u) {
      return rund::AccelCheck{false, "accel_metal_command_unavailable"};
    }
    id<MTLDevice> const device =
        (__bridge id<MTLDevice>)sequence->adapter->device.get();
    std::shared_ptr<MetalResidencyCandidate> candidate;
    try {
      candidate = std::make_shared<MetalResidencyCandidate>();
    } catch (const std::bad_alloc &) {
      return rund::AccelCheck{false, "accel_metal_command_unavailable"};
    }
    candidate->sequence = sequence;
    MetalResidencySubmission &submission = candidate->submission;
    submission.allocator = [device newCommandAllocator];
    submission.command = [device newCommandBuffer];
    submission.event = [device newSharedEvent];
    submission.listener = [MTLSharedEventListener sharedListener];
    submission.owner = std::weak_ptr<void>{prepared};
    submission.watchdog = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_TIMER, 0u, 0u,
        dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0u));
    MetalResidencySlidingGate &sliding = candidate->sliding;
    static_cast<void>(PrepareMetalResidencySlidingGate(*sequence, sliding));
    MetalResidencyWindow &window = candidate->window;
    window.ready = [device newSharedEvent];
    window.done = [device newSharedEvent];
    window.listener = [MTLSharedEventListener sharedListener];
    window.service = dispatch_queue_create("rund.metal.residency.service",
                                           DISPATCH_QUEUE_SERIAL);
    window.owner = std::weak_ptr<void>{prepared};
    for (MetalResidencyWindowCommand &command : window.commands) {
      command.allocator = [device newCommandAllocator];
      command.command = [device newCommandBuffer];
      command.watchdog = dispatch_source_create(
          DISPATCH_SOURCE_TYPE_TIMER, 0u, 0u,
          dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0u));
    }
    MTLResidencySetDescriptor *const descriptor =
        [MTLResidencySetDescriptor new];
    descriptor.initialCapacity =
        sequence->warm.resource_count + sequence->warm.chunk_count +
        sequence->pipelines.size() +
        static_cast<NSUInteger>(sliding.ready_for_submit ? 3u : 0u);
    NSError *error = nil;
    submission.resources = [device newResidencySetWithDescriptor:descriptor
                                                           error:&error];
    if (sequence->adapter->residency_queue == nullptr ||
        submission.allocator == nil || submission.command == nil ||
        submission.event == nil || submission.listener == nil ||
        submission.watchdog == nil || submission.resources == nil) {
      return rund::AccelCheck{false, "accel_metal_command_unavailable"};
    }
    if (window.ready == nil || window.done == nil || window.listener == nil ||
        window.service == nil) {
      return rund::AccelCheck{false, "accel_metal_command_unavailable"};
    }
    for (const MetalResidencyWindowCommand &command : window.commands) {
      if (command.allocator == nil || command.command == nil ||
          command.watchdog == nil) {
        return rund::AccelCheck{false, "accel_metal_command_unavailable"};
      }
    }
    const std::weak_ptr<void> terminal_owner = submission.owner;
    dispatch_source_set_event_handler(submission.watchdog, ^{
      const std::shared_ptr<void> retained = terminal_owner.lock();
      if (retained == nullptr) {
        return;
      }
      auto &target = *static_cast<MetalSequence *>(retained.get());
      MetalResidencySubmission &active = target.residency_submission;
      const std::uint64_t generation = active.terminal.generation();
      const std::uint64_t deadline =
          active.watchdog_deadline_ns.load(std::memory_order_acquire);
      if (generation == 0u || deadline == 0u) {
        return;
      }
      const std::uint64_t now = MonotonicNanoseconds();
      if (now < deadline) {
        dispatch_source_set_timer(
            active.watchdog,
            dispatch_time(DISPATCH_TIME_NOW,
                          static_cast<int64_t>(deadline - now)),
            DISPATCH_TIME_FOREVER, 0u);
        return;
      }
      CompleteMetalResidencySubmission(target, generation,
                                       NativeTerminal::UnknownMayWrite);
    });
    dispatch_resume(submission.watchdog);
    dispatch_source_set_timer(submission.watchdog, DISPATCH_TIME_FOREVER,
                              DISPATCH_TIME_FOREVER, 0u);
    const std::weak_ptr<void> window_owner = window.owner;
    for (std::size_t slot = 0u; slot < window.commands.size(); ++slot) {
      MetalResidencyWindowCommand &command = window.commands[slot];
      dispatch_source_set_event_handler(command.watchdog, ^{
        const std::shared_ptr<void> retained = window_owner.lock();
        if (retained == nullptr) {
          return;
        }
        auto &target = *static_cast<MetalSequence *>(retained.get());
        MetalResidencyWindowCommand &active =
            target.residency_window.commands[slot];
        const std::uint64_t generation = active.terminal.generation();
        const std::uint64_t deadline =
            active.watchdog_deadline_ns.load(std::memory_order_acquire);
        if (generation == 0u || deadline == 0u) {
          return;
        }
        const std::uint64_t now = MonotonicNanoseconds();
        if (now < deadline) {
          dispatch_source_set_timer(
              active.watchdog,
              dispatch_time(DISPATCH_TIME_NOW,
                            static_cast<int64_t>(deadline - now)),
              DISPATCH_TIME_FOREVER, 0u);
          return;
        }
        CompleteMetalResidencyWindowCommand(target, slot, generation,
                                            NativeTerminal::UnknownMayWrite);
      });
      dispatch_resume(command.watchdog);
      dispatch_source_set_timer(command.watchdog, DISPATCH_TIME_FOREVER,
                                DISPATCH_TIME_FOREVER, 0u);
    }
    [submission.resources
        addAllocations:(const id<MTLAllocation> *)sequence->warm.resources
                 count:sequence->warm.resource_count];
    for (NSUInteger index = 0u; index < sequence->warm.chunk_count; ++index) {
      const MetalIcbChunk &chunk = sequence->warm.chunks[index];
      if (!chunk.valid()) {
        return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
      }
      [submission.resources addAllocation:chunk.commands];
    }
    for (id<MTLComputePipelineState> const &pipeline : sequence->pipelines) {
      if (pipeline == nil) {
        return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
      }
      [submission.resources addAllocation:pipeline];
    }
    if (sliding.ready_for_submit) {
      [submission.resources addAllocation:sliding.descriptor];
      [submission.resources addAllocation:sliding.pipeline];
      [submission.resources addAllocation:sliding.command];
    }
    [submission.resources commit];
    if (submission.resources.allocationCount != descriptor.initialCapacity) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    if (!MetalResidencyRetainedBytes(submission, sliding, window,
                                     retained_bytes)) {
      return rund::AccelCheck{false, "accel_metal_command_unavailable"};
    }
    submission.phase = MetalResidencySubmissionPhase::Cold;
    window.phase = MetalResidencyWindowPhase::Cold;
    candidate->retained_bytes = retained_bytes;
    owner = std::move(candidate);
  }
#endif
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
