#include "local.hpp"

#include <rund/counter.hpp>

#include <cassert>
#include <limits>
#include <memory>
#include <new>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
struct MetalResidencyCandidate final {
  MetalSequence *sequence{};
  MetalResidencySubmission submission{};
  std::uint64_t retained_bytes{};
};

[[nodiscard]] bool
MetalResidencyRetainedBytes(const MetalResidencySubmission &submission,
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
  return true;
}
#endif

} // namespace

MetalSequence::~MetalSequence() {
#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  if (@available(macOS 26.0, iOS 26.0, *)) {
    MetalResidencySubmission &submission = residency_submission;
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
    submission.allocator = nil;
    submission.queue = nil;
  }
#endif
}

rund::AccelCheck
StageMetalPipelineResidency(const std::shared_ptr<void> &prepared,
                            std::shared_ptr<void> &owner,
                            std::uint64_t &retained_bytes) noexcept {
  owner.reset();
  retained_bytes = 0u;
  auto *const sequence = static_cast<MetalSequence *>(prepared.get());
  if (!ValidMetalSequence(sequence) ||
      sequence->residency_submission.supported()) {
    return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
  }
#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  if (@available(macOS 26.0, iOS 26.0, *)) {
    if (sequence->adapter == nullptr || sequence->warm.resources == nullptr ||
        sequence->warm.resource_count == 0u ||
        sequence->warm.chunks == nullptr || sequence->warm.chunk_count == 0u ||
        sequence->memory_meter == nullptr ||
        sequence->warm.resource_count > std::numeric_limits<NSUInteger>::max() -
                                            sequence->warm.chunk_count) {
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
    submission.queue = [device newMTL4CommandQueue];
    submission.allocator = [device newCommandAllocator];
    submission.command = [device newCommandBuffer];
    submission.event = [device newSharedEvent];
    MTLResidencySetDescriptor *const descriptor =
        [MTLResidencySetDescriptor new];
    descriptor.initialCapacity =
        sequence->warm.resource_count + sequence->warm.chunk_count;
    NSError *error = nil;
    submission.resources = [device newResidencySetWithDescriptor:descriptor
                                                           error:&error];
    if (submission.queue == nil || submission.allocator == nil ||
        submission.command == nil || submission.event == nil ||
        submission.resources == nil) {
      return rund::AccelCheck{false, "accel_metal_command_unavailable"};
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
    [submission.resources commit];
    if (!MetalResidencyRetainedBytes(submission, retained_bytes)) {
      return rund::AccelCheck{false, "accel_metal_command_unavailable"};
    }
    submission.phase = MetalResidencySubmissionPhase::Cold;
    candidate->retained_bytes = retained_bytes;
    owner = std::move(candidate);
  }
#endif
  return rund::AccelCheck{true, "ok"};
}

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
    sequence->residency_submission = candidate->submission;
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

void PrimeMetalResidencySubmission(MetalSequence &sequence,
                                   const bool succeeded) noexcept {
  MetalResidencySubmission &submission = sequence.residency_submission;
  if (submission.phase != MetalResidencySubmissionPhase::Cold) {
    return;
  }
  submission.phase = succeeded ? MetalResidencySubmissionPhase::Ready
                               : MetalResidencySubmissionPhase::Failed;
}

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
    supported = true;
  }
#endif
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
