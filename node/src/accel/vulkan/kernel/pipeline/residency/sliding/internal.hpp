#pragma once

#include "../local.hpp"

#include "../../../../../clock.hpp"
#include "../../../../command.hpp"
#include "../../../../command/resources.hpp"
#include "../../../../descriptor.hpp"
#include "../../../../runtime/counter.hpp"
#include "../../../control/source.hpp"
#include "../../../lease.hpp"
#include "../../source/artifact.hpp"
#include "../../state.hpp"

#include <rund/compute/reason.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

namespace rund::node::accel::detail::vulkan_sliding_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

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

static_assert(ControlGenerationWord < VulkanResidencySlidingLocalWord);
static_assert(VulkanResidencySlidingLocalWord + ResidencyWindowLocalCapacity ==
              VulkanResidencySlidingRowWords);

[[nodiscard]] std::string_view gate_source() noexcept;

[[nodiscard]] bool coherent_storage(const VulkanBuffer &buffer) noexcept;
[[nodiscard]] bool
build_payload(VulkanPipeline &pipeline, VulkanResidencySlidingGate &gate,
              const BackendResidencySlidingDescriptor &descriptor,
              std::span<const std::uint32_t> locals,
              VulkanResidencySlidingPayload &payload) noexcept;
void commit_descriptor(VulkanResidencySlidingGate &gate,
                       const BackendResidencySlidingDescriptor &row) noexcept;
void observe_submit_frontier(VulkanResidencySlidingGate &gate) noexcept;
void pause_terminal_frontier(VulkanResidencySlidingGate &gate) noexcept;

[[nodiscard]] bool
record_gate_command(VulkanPipeline &pipeline,
                    VulkanResidencySelection &selection) noexcept;
[[nodiscard]] bool
create_ready_semaphore(VulkanPipeline &pipeline,
                       VulkanResidencySlidingGate &gate) noexcept;
[[nodiscard]] bool record_generated_map_role(
    VulkanPipeline &pipeline, VulkanResidencySelection &selection,
    const PersistentResidencySlidingRole &source, std::uint32_t stride,
    std::uint64_t plan_identity, std::uint64_t token,
    std::uint64_t run_generation, std::span<const std::uint32_t> locals,
    std::span<VkCommandBuffer> commands, std::size_t &command_count,
    std::uint64_t &dispatch_count, std::uint64_t &control_count,
    std::uint64_t &reset_count, std::uint64_t &reset_bytes,
    VulkanResidencyPersistentRun *expected_run, bool tail = false) noexcept;
[[nodiscard]] bool
queue_gate_submission(VulkanAdapter &adapter, VulkanResidencySlidingGate &gate,
                      std::span<const VkCommandBuffer> commands, VkFence fence,
                      KernelCompletion completion, void *user,
                      bool collect_timing) noexcept;

void complete(void *raw, KernelResult result) noexcept;

#endif

} // namespace rund::node::accel::detail::vulkan_sliding_detail
