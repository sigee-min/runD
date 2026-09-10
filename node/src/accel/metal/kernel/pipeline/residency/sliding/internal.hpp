#pragma once

#include "../local.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::node::accel::detail::metal_residency_sliding {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK) &&                 \
    defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0

inline constexpr std::size_t OwnerWord = 0u;
inline constexpr std::size_t PlanWord = 2u;
inline constexpr std::size_t TokenWord = 4u;
inline constexpr std::size_t RunWord = 6u;
inline constexpr std::size_t CoordinateWord = 8u;
inline constexpr std::size_t TurnWord = 10u;
inline constexpr std::size_t ReadMaskWord = 12u;
inline constexpr std::size_t WriteMaskWord = 14u;
inline constexpr std::size_t DescriptorGenerationWord = 16u;
inline constexpr std::size_t GenerationStrideWord = 18u;
inline constexpr std::size_t ArgumentCountWord = 19u;
inline constexpr std::size_t LocalCountWord = 20u;
inline constexpr std::size_t StrideWord = 21u;
inline constexpr std::size_t SlotWord = 22u;
inline constexpr std::size_t InvalidReasonWord = 23u;
inline constexpr std::size_t StepCountWord = 24u;
inline constexpr std::size_t ControlGenerationWord = 25u;

static_assert(ControlGenerationWord < MetalResidencySlidingLocalWord);
static_assert(MetalResidencySlidingLocalWord + ResidencyWindowLocalCapacity ==
              MetalResidencySlidingRowWords);

inline constexpr std::uint64_t TerminalDeadlineNs = 30u * NSEC_PER_SEC;
inline constexpr std::uint64_t FaultTerminalDeadlineNs = 10u * NSEC_PER_MSEC;

[[nodiscard]] const char *GateSource() noexcept;

[[nodiscard]] bool BuildPayload(MetalSequence &, MetalResidencySlidingGate &,
                                const BackendResidencySlidingDescriptor &,
                                std::span<const std::uint32_t>,
                                MetalResidencySlidingPayload &) noexcept;

void CommitDescriptor(MetalResidencySlidingGate &,
                      const BackendResidencySlidingDescriptor &) noexcept;

#endif

} // namespace rund::node::accel::detail::metal_residency_sliding
