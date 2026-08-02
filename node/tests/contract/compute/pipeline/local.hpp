#pragma once

#include <rund/compute.hpp>
#include <rund/compute/pipeline.hpp>

#include <array>
#include <cstdint>
#include <span>

namespace rund_node_test_pipeline {

using rund::compute::Backend;
using rund::compute::Buffer;
using rund::compute::Fixed;
using rund::compute::Pipeline;
using rund::compute::PipelineBuilder;
using rund::compute::Reason;
using rund::compute::StateSnapshot;
using rund::compute::Stats;

template <class T, std::size_t N>
[[nodiscard]] bool ReadExact(const Pipeline &pipeline, const Buffer<T> &buffer,
                             std::array<T, N> &output) {
  return static_cast<bool>(pipeline.read(buffer, std::span<T>{output}));
}

template <class T, std::size_t N>
[[nodiscard]] auto Upload(rund::compute::Device &device,
                          const std::array<T, N> &values) {
  return device.upload<T>(std::span<const T>{values});
}

[[nodiscard]] bool Overwrite(Buffer<std::uint32_t> &,
                             std::span<const std::uint32_t>);

template <std::size_t N>
[[nodiscard]] bool Overwrite(Buffer<std::uint32_t> &buffer,
                             const std::array<std::uint32_t, N> &values) {
  return Overwrite(buffer, std::span<const std::uint32_t>{values});
}

[[nodiscard]] bool WarmCountersClean(const Stats &) noexcept;
[[nodiscard]] bool
CanonicalChunkOrder(const rund::compute::detail::ProgramState &) noexcept;
[[nodiscard]] bool
SameControlStats(const rund::compute::ControlStats &,
                 const rund::compute::ControlStats &) noexcept;
[[nodiscard]] bool ProfileMemoryReconciles(
    const rund::compute::PipelineProfileSnapshot &,
    std::span<const rund::compute::PipelineStepProfile>) noexcept;
[[nodiscard]] bool SameWarmStats(const rund::compute::Stats &,
                                 const rund::compute::Stats &) noexcept;
[[nodiscard]] bool
    SameMemoryEntries(std::span<const rund::compute::MemoryEntry>,
                      std::span<const rund::compute::MemoryEntry>) noexcept;
[[nodiscard]] bool SameMemory(const rund::compute::MemoryStats &,
                              const rund::compute::MemoryStats &) noexcept;
[[nodiscard]] bool
TimingUnavailable(const rund::compute::StepTiming &) noexcept;

[[nodiscard]] int CheckRepeat(rund::compute::Device &, Backend);
[[nodiscard]] int CheckIterationHistory(rund::compute::Device &, Backend);
[[nodiscard]] int CheckHostFeedback(rund::compute::Device &, Backend);
[[nodiscard]] int CheckSealedRepetitions(rund::compute::Device &, Backend,
                                         rund::compute::graph::Fingerprint &);
[[nodiscard]] int CheckSurface(rund::compute::Device &);
[[nodiscard]] int CheckWideFixed(rund::compute::Device &, Backend,
                                 rund::compute::graph::Fingerprint &,
                                 std::uint64_t &);
[[nodiscard]] int CheckMemory(rund::compute::Device &);
[[nodiscard]] int CheckDevicePipelineMemoryAdmission();
[[nodiscard]] int CheckMetalGuardTransform();
[[nodiscard]] int CheckViews(rund::compute::Device &, Backend);
[[nodiscard]] int CheckViewArena(rund::compute::Device &, Backend);
[[nodiscard]] int CheckHazards(rund::compute::Device &);
[[nodiscard]] int CheckValidationAndIdentity(rund::compute::Device &);
[[nodiscard]] int CheckTransactionalGenerations(rund::compute::Device &,
                                                Backend, std::uint64_t &,
                                                std::uint64_t &);
[[nodiscard]] int CheckReusableCheckpoints(rund::compute::Device &, Backend);
[[nodiscard]] int CheckVulkanCheckpointChunking(rund::compute::Device &,
                                                Backend);
[[nodiscard]] int CheckNativeDeviceLoss(rund::compute::Device &, Backend);
[[nodiscard]] int CheckUnknownCompletionProfileIdentity(rund::compute::Device &,
                                                        Backend);
[[nodiscard]] int CheckProfile(rund::compute::Device &, Backend);
[[nodiscard]] int CheckZeroWork(rund::compute::Device &, Backend);
[[nodiscard]] int CheckFrozenCpuMapBindings(rund::compute::Device &);
[[nodiscard]] int CheckSemanticStatus(rund::compute::Device &, Backend);
[[nodiscard]] int CheckBackend(Backend, rund::compute::graph::Fingerprint &,
                               rund::compute::graph::Fingerprint &,
                               std::uint64_t &, std::uint64_t &,
                               std::uint64_t &);

} // namespace rund_node_test_pipeline
