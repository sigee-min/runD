#pragma once

#include "../../../../kernel/residency/persistent_sliding.hpp"
#include "../../../../kernel/residency/sliding.hpp"
#include "../../../kernel.hpp"
#include "../state.hpp"

#include <span>

namespace rund::node::accel::detail {

struct PreparedResidencyPersistentSlidingRole;

[[nodiscard]] PersistentResidencySlidingCapability
QueryMetalPreparedPersistentSlidingCapability(
    std::span<const PreparedResidencyPersistentSlidingRole>, std::uint64_t,
    ResidencySlidingMemory, PersistentResidencySlidingMode) noexcept;

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

[[nodiscard]] rund::AccelCheck
MetalPipelineResidencyReady(const std::shared_ptr<void> &prepared,
                            bool &ready) noexcept;

[[nodiscard]] BackendResidencySlidingCapability
MetalResidencySlidingCapability(const std::shared_ptr<void> &prepared,
                                ResidencySlidingMemory memory) noexcept;
[[nodiscard]] rund::AccelCheck SubmitMetalResidencySliding(
    const std::shared_ptr<void> &prepared,
    const BackendResidencySlidingDescriptor &, KernelCompletion, void *,
    KernelTiming, PipelineSubmitMode, std::span<const std::uint32_t>) noexcept;
[[nodiscard]] rund::AccelCheck
PrepareMetalResidencySlidingGate(MetalSequence &,
                                 MetalResidencySlidingGate &) noexcept;
// Called with the adapter terminal gate held.  It classifies acquired GPU
// evidence, publishes quarantine before a peer can enter the queue, and
// returns the strong owner that must survive the later unlocked callback.
[[nodiscard]] std::shared_ptr<void>
FinalizeMetalResidencySliding(MetalSequence &, KernelResult &) noexcept;
void CompleteMetalSequence(void *, KernelResult) noexcept;
[[nodiscard]] bool InjectMetalResidencySlidingStaleDescriptorOnce(
    const std::shared_ptr<void> &) noexcept;

// Metal-private cold lowering for the backend-neutral persistent-product
// service seam. The preparation owns one legacy command buffer containing all
// Q event dependencies and fixed role executions; no service operation may
// encode or commit native work.
[[nodiscard]] rund::AccelCheck SubmitMetalResidencySubmission(
    MetalSequence &sequence, KernelTiming timing,
    std::span<const std::uint32_t> locals = {}) noexcept;

#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
[[nodiscard]] id<MTL4CommandQueue>
MetalResidencyScheduleQueue(const MetalSequence &) noexcept
    API_AVAILABLE(macos(26.0), ios(26.0));

[[nodiscard]] rund::AccelCheck EncodeMetalResidencySlidingCommand(
    MetalSequence &, std::span<const std::uint32_t>,
    std::uint64_t &dispatch_count, std::uint64_t &control_count,
    std::uint64_t &reset_count, std::uint64_t &reset_bytes) noexcept
    API_AVAILABLE(macos(26.0), ios(26.0));

[[nodiscard]] rund::AccelCheck EncodeMetalResidencyScheduleCommand(
    MetalSequence &, id<MTL4CommandAllocator>, id<MTL4CommandBuffer>,
    std::span<const std::uint32_t>, std::uint64_t &dispatch_count,
    std::uint64_t &control_count, std::uint64_t &reset_count,
    std::uint64_t &reset_bytes) noexcept API_AVAILABLE(macos(26.0), ios(26.0));

[[nodiscard]] bool ObserveMetalResidencyScheduleEvidence(
    MetalSequence &, PreparedPipelineBackendEvidence &) noexcept
    API_AVAILABLE(macos(26.0), ios(26.0));

void CompleteMetalResidencySubmission(MetalSequence &sequence,
                                      std::uint64_t generation,
                                      NativeTerminal terminal) noexcept
    API_AVAILABLE(macos(26.0), ios(26.0));
void CompleteMetalResidencyWindowCommand(MetalSequence &sequence,
                                         std::size_t slot,
                                         std::uint64_t generation,
                                         NativeTerminal terminal) noexcept
    API_AVAILABLE(macos(26.0), ios(26.0));
#endif

#endif

} // namespace rund::node::accel::detail
