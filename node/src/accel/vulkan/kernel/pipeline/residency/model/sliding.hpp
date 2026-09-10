#pragma once

#include "base.hpp"

#include <memory>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

// Word-addressed std430 layout keeps authentication independent of Host
// compiler padding while preserving every 64-bit identity exactly.
inline constexpr std::size_t VulkanResidencySlidingRowWords = 58u;
inline constexpr std::size_t VulkanResidencySlidingLocalWord = 26u;
inline constexpr std::size_t VulkanResidencySlidingPayloadWords = 120u;
inline constexpr std::size_t VulkanResidencySlidingPublishedWord =
    VulkanResidencySlidingRowWords;
inline constexpr std::size_t VulkanResidencySlidingAcceptedWord =
    VulkanResidencySlidingRowWords * 2u;
inline constexpr std::size_t VulkanResidencySlidingReasonWord =
    VulkanResidencySlidingAcceptedWord + 1u;
inline constexpr std::size_t VulkanResidencySlidingObservedGenerationWord =
    VulkanResidencySlidingReasonWord + 1u;

struct VulkanResidencySlidingPayload final {
  std::array<std::uint32_t, VulkanResidencySlidingPayloadWords> words{};
};

static_assert(sizeof(VulkanResidencySlidingPayload) == 480u);

struct VulkanResidencySlidingSubmissionStorage final {
  std::array<VkCommandBuffer, PreparedPipelineStepCapacity + 3u> commands{};
  VulkanResidencySlidingPayload payload{};
};

struct VulkanResidencyGraphSubmitPlan final {
  std::array<VkCommandBuffer, PreparedPipelineStepCapacity + 2u> commands{};
  std::array<bool, PreparedPipelineStepCapacity> local_mask{};
  std::size_t command_count{};
  std::size_t active_count{};
  std::uint64_t next_submit_seq{};
  std::uint64_t next_submit_total{};
  bool valid{};
};

// Cold, fixed per-selection resources. The gate authenticates an already
// selected coordinate and writes only its retained indirect argument arena.
struct VulkanResidencySlidingGate final {
  VulkanCommand command{};
  std::array<std::array<VulkanCommand, PreparedPipelineStepCapacity>,
             ResidencySlidingCapacity>
      generated_roles{};
  std::array<VulkanCommand, PreparedPipelineStepCapacity> generated_tail{};
  VulkanBuffer descriptor{};
  VulkanBuffer original_arguments{};
  VulkanBuffer argument_owners{};
  VulkanCollectivePipeline *pipeline{};
  VkDescriptorSet descriptor_set{VK_NULL_HANDLE};
  std::vector<VulkanCollectiveDescriptorLease> descriptor_leases;
  VkSemaphore ready{VK_NULL_HANDLE};
  std::uint64_t next_ready_value{1u};
  const void *owner{};
  std::uint64_t plan_identity{};
  std::uint64_t token{};
  std::uint64_t run_generation{};
  std::uint64_t last_coordinate{};
  std::uint64_t last_turn{};
  std::uint64_t last_descriptor_generation{};
  std::uint8_t stride{};
  std::uint8_t slot{};
  std::uint64_t retained_bytes{};
  std::array<std::uint32_t, VulkanResidencySlidingRowWords> published_row{};
  std::shared_ptr<void> active_owner{};
  std::atomic_bool force_stale_once{false};
  std::atomic_bool pause_terminal_frontier_once{false};
  std::atomic_bool terminal_frontier_release{true};
  std::atomic<std::uint64_t> submit_frontier_count{};
  std::atomic<std::uint64_t> terminal_frontier_count{};
  std::atomic<std::uint64_t> gpu_result_read_count{};
  bool authenticated{};
  bool ready_for_submit{};
  bool map_ready{};
  std::atomic_bool quarantined{};
};

#endif

} // namespace rund::node::accel::detail
