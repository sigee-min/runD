#pragma once

#include "../internal.hpp"

#include <mutex>

namespace rund::node::accel::detail::metal_residency_sliding::submit {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK) &&                 \
    defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0

struct Context final {
  const std::shared_ptr<void> &prepared;
  const BackendResidencySlidingDescriptor &descriptor;
  std::span<const std::uint32_t> locals{};
  MetalSequence &sequence;
  submission::State<MetalSequence> &claim;
  MetalAdapter &adapter;
  MetalResidencySubmission &native;
  MetalResidencySlidingGate &gate;
  id<MTL4CommandQueue> queue = nil;
  void *guard{};
  void *destination{};
  MetalResidencySlidingSubmissionStorage storage{};
  std::span<const std::uint32_t> authenticated_locals{};
  std::uint64_t submit_begin{};
  std::uint64_t terminal_generation{};
  std::uint64_t ready_value{};
};

[[nodiscard]] rund::AccelCheck Begin(const std::shared_ptr<void> &,
                                     KernelCompletion, void *, KernelTiming,
                                     PipelineSubmitMode,
                                     MetalSequence *&) noexcept;

[[nodiscard]] Context Capture(const std::shared_ptr<void> &,
                              const BackendResidencySlidingDescriptor &,
                              std::span<const std::uint32_t>,
                              MetalSequence &) noexcept
    API_AVAILABLE(macos(26.0), ios(26.0));

[[nodiscard]] rund::AccelCheck Validate(const Context &) noexcept;

[[nodiscard]] rund::AccelCheck Cancel(Context &, std::unique_lock<std::mutex> &,
                                      rund::AccelCheck) noexcept;

[[nodiscard]] rund::AccelCheck Build(Context &) noexcept;

[[nodiscard]] rund::AccelCheck Encode(Context &) noexcept
    API_AVAILABLE(macos(26.0), ios(26.0));

[[nodiscard]] rund::AccelCheck ArmTerminal(Context &) noexcept
    API_AVAILABLE(macos(26.0), ios(26.0));

void Activate(Context &) noexcept;

void InstallTerminalListener(Context &) noexcept
    API_AVAILABLE(macos(26.0), ios(26.0));

void PublishDescriptor(Context &) noexcept;

void CommitQueue(Context &) noexcept API_AVAILABLE(macos(26.0), ios(26.0));

void ArmWatchdog(Context &) noexcept;

void SignalKnownTerminal(Context &) noexcept
    API_AVAILABLE(macos(26.0), ios(26.0));

#endif

} // namespace rund::node::accel::detail::metal_residency_sliding::submit
