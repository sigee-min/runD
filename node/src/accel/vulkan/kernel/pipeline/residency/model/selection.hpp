#pragma once

#include "graph.hpp"
#include "sliding.hpp"

#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

// Cold-recorded owner table. Warm submission borrows command handles from
// this immutable selection and never reconstructs residency policy.
struct VulkanResidencySelection final {
  ~VulkanResidencySelection();

  VulkanAdapter *adapter{};
  VulkanCommand prefix{};
  std::vector<VulkanResidencyStep> steps;
  VulkanCommand suffix{};
  VulkanBuffer arguments{};
  std::vector<VkDispatchIndirectCommand> original_arguments;
  std::vector<std::uint32_t> argument_owners;
  VulkanResidencySlidingGate sliding{};
  VulkanResidencyWindowRun window{};
  std::atomic<VulkanResidencyWindowRun *> active_window{};
  VulkanResidencyScheduleRun schedule{};
  std::atomic<VulkanResidencyScheduleRun *> active_schedule{};
  VulkanResidencyPersistentRun persistent{};
  std::atomic<VulkanResidencyPersistentRun *> active_persistent{};
  std::uint64_t host_bytes{};
  VulkanResidencyMode mode{VulkanResidencyMode::Direct};
  VulkanResidencyGraphStageDirectProof graph_direct{};
  VulkanResidencyGraphStageGeneratedProof graph_generated{};
  std::array<VulkanResidencyGraphGeneratedFrame, PreparedPipelineStepCapacity>
      graph_generated_frames{};
  std::array<bool, PreparedPipelineStepCapacity> graph_generated_submitted{};
  std::size_t graph_generated_submitted_count{};
  VulkanResidencyGraphGeneratedPhase graph_generated_phase{
      VulkanResidencyGraphGeneratedPhase::NotApplicable};
  VulkanResidencyGraphGeneratedDiagnostics graph_generated_diagnostics{};
  std::uint64_t graph_generated_generation{};
  std::size_t graph_generated_command_count{};
  VulkanResidencyGraphStageSequenceProof graph_sequence{};
  std::array<VulkanResidencyGraphSequenceFrame, PreparedPipelineStepCapacity>
      graph_sequence_frames{};
  std::array<bool, PreparedPipelineStepCapacity> graph_sequence_submitted{};
  std::size_t graph_sequence_submitted_count{};
  std::size_t graph_sequence_command_count{};
  std::uint64_t submit_seq{};
  std::uint64_t done_seq{};
  std::uint64_t submit_total{};
  std::uint64_t done_total{};
  std::uint64_t accept_total{};
  std::uint64_t known_total{};
  std::uint64_t unknown_total{};
  std::atomic_bool quarantined{};
  bool bounded{};
  bool ready{};
};

#endif

} // namespace rund::node::accel::detail
