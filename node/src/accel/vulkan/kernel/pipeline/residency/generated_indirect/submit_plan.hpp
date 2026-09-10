#pragma once

#include "../local.hpp"
#include "../mode.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace rund::node::accel::detail::vulkan_generated_indirect_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

// The plan is an immutable queue-admission value.  It contains only the
// command handles and checked monotonic values needed by the commit owner;
// selection and lifecycle state remain owned by VulkanResidencySelection.
struct Plan final {
  bool generated{};
  VulkanResidencyGraphSubmitPlan graph{};
};

[[nodiscard]] rund::AccelCheck
select(const VulkanPipeline &pipeline, std::span<const std::uint32_t> locals,
       std::span<VkCommandBuffer> commands, std::size_t &command_count,
       std::uint64_t &dispatch_count, std::uint64_t &control_count,
       Plan &plan) noexcept;

[[nodiscard]] rund::AccelCheck
build_submit_plan(const VulkanPipeline &pipeline,
                  std::span<const std::uint32_t> locals,
                  VulkanResidencyGraphSubmitPlan &plan) noexcept;

namespace impl {

[[nodiscard]] constexpr std::size_t
rows_per_local(const VulkanResidencyMode mode) noexcept {
  return mode == VulkanResidencyMode::GraphStageSequence ? 2u : 1u;
}

[[nodiscard]] inline bool active_rows(const VulkanResidencyMode mode,
                                      const std::size_t active_count,
                                      std::size_t &row_count) noexcept {
  row_count = 0u;
  if ((mode != VulkanResidencyMode::GraphStageGeneratedIndirect &&
       mode != VulkanResidencyMode::GraphStageSequence) ||
      active_count == 0u || active_count > PreparedPipelineStepCapacity) {
    return false;
  }
  const std::size_t rows = rows_per_local(mode);
  if (active_count > std::numeric_limits<std::size_t>::max() / rows) {
    return false;
  }
  row_count = active_count * rows;
  return true;
}

[[nodiscard]] bool graph_active_rows(std::span<const std::uint32_t> locals,
                                     std::size_t capacity,
                                     VulkanResidencyMode mode,
                                     std::size_t &active_count,
                                     std::size_t &row_count) noexcept;

[[nodiscard]] bool build_plan(const VulkanPipeline &pipeline,
                              std::span<const std::uint32_t> locals,
                              VulkanResidencyGraphSubmitPlan &plan) noexcept;

} // namespace impl

#endif

} // namespace rund::node::accel::detail::vulkan_generated_indirect_detail
