#pragma once

#include "admission.hpp"

#include <kernel/program/compute/graph/schema.hpp>

#include <memory>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanResidencyGraphStageGeneratedProof final {
  static constexpr std::size_t GraphGeneratedDataBindingCapacity = 8u;

  std::uint64_t kernel_id{};
  std::uint64_t graph_id_hi{};
  std::uint64_t graph_id_lo{};
  std::uint64_t node_count{};
  std::uint64_t op_hash_hi{};
  std::uint64_t op_hash_lo{};
  std::array<std::uint64_t, GraphGeneratedDataBindingCapacity> binding_ids{};
  std::array<std::uint64_t, GraphGeneratedDataBindingCapacity> binding_bytes{};
  std::array<std::uint64_t, GraphGeneratedDataBindingCapacity>
      binding_offsets{};
  std::array<std::uint64_t, GraphGeneratedDataBindingCapacity>
      binding_elements{};
  std::array<std::uint64_t, GraphGeneratedDataBindingCapacity>
      binding_strides{};
  std::array<std::uint64_t, GraphGeneratedDataBindingCapacity> binding_counts{};
  std::array<std::uint8_t, GraphGeneratedDataBindingCapacity> binding_roles{};
  std::array<std::uint64_t, GraphGeneratedDataBindingCapacity>
      binding_logical_elements{};
  std::array<std::uint64_t, GraphGeneratedDataBindingCapacity>
      binding_logical_strides{};
  std::array<std::uint64_t, GraphGeneratedDataBindingCapacity>
      binding_logical_counts{};
  std::array<std::shared_ptr<void>, GraphGeneratedDataBindingCapacity>
      binding_handles{};
  std::uint32_t binding_count{};
  std::uint32_t read_count{};
  std::uint32_t write_count{};
  std::uint32_t local_count{};
  std::uint32_t count_binding{rund::kernel::kNoGraphControlBinding};
  std::uint32_t predicate_binding{rund::kernel::kNoGraphControlBinding};
  std::uint64_t count_offset{};
  std::uint64_t predicate_offset{};
  std::uint64_t count_capacity{};
  std::uint64_t predicate_expected{};
  std::uint64_t count_id{};
  std::uint64_t predicate_id{};
  std::uint64_t count_bytes{};
  std::uint64_t predicate_bytes{};
  std::shared_ptr<void> count_handle{};
  std::shared_ptr<void> predicate_handle{};
  std::uint8_t count_source{};
  std::uint8_t predicate_source{};
  std::uint32_t frame_index{};
  std::uint64_t gate_generation{};
  VulkanResidencyGraphGeneratedPredicate first_failure_predicate{
      VulkanResidencyGraphGeneratedPredicate::None};
  std::uint32_t first_failure_ordinal{VulkanResidencyAdmissionNoOrdinal};
  const char *first_failure_reason{"not_evaluated"};
  bool has_count{};
  bool has_predicate{};
  bool valid{};
};

struct VulkanResidencyGraphGeneratedFrame final {
  VulkanResidencyGraphStageGeneratedProof proof{};
  VulkanCommand command{};
  VulkanBuffer gate{};
  VulkanCollectivePipeline *control_pipeline{};
  VkDescriptorSet control_descriptor{VK_NULL_HANDLE};
  std::uint64_t generation{};
  std::uint64_t expected_descriptor_generation{};
  bool ready{};
  bool submitted{};
  bool quarantined{};
};

struct VulkanResidencyGraphStageSequenceProof final {
  std::array<VulkanResidencyGraphStageGeneratedProof, 2u> steps{};
  std::array<std::array<VulkanResidencyGraphStageGeneratedProof, 2u>, 2u>
      frame_proofs{};
  std::uint32_t stage_count{};
  std::uint32_t local_count{};
  VulkanResidencyGraphGeneratedPredicate first_failure_predicate{
      VulkanResidencyGraphGeneratedPredicate::None};
  std::uint32_t first_failure_ordinal{VulkanResidencyAdmissionNoOrdinal};
  const char *first_failure_reason{"not_evaluated"};
  bool valid{};
};

struct VulkanResidencyGraphSequenceFrame final {
  VulkanCommand command{};
  std::array<VulkanResidencyGraphStageGeneratedProof, 2u> proofs{};
  std::array<VulkanBuffer, 2u> gates{};
  std::array<VulkanCollectivePipeline *, 2u> control_pipelines{};
  std::array<VkDescriptorSet, 2u> control_descriptors{};
  std::array<std::uint64_t, 2u> generations{};
  std::array<std::shared_ptr<void>, 2u> intermediate_pins{};
  std::array<bool, 2u> ready{};
  bool command_ready{};
  bool submitted{};
  bool quarantined{};
};

// The common graph authority owns ports, pages, aliases, and selected locals;
// this proof freezes only the collision-safe backend command-table identity.
struct VulkanResidencyGraphStageDirectProof final {
  std::uint64_t kernel_id{};
  std::uint64_t graph_id_hi{};
  std::uint64_t graph_id_lo{};
  std::uint64_t node_count{};
  std::uint32_t operation{};
  std::uint32_t read_count{};
  std::uint32_t write_count{};
  std::uint32_t local_count{};
  std::uint64_t direct_dispatch_count{};
  bool valid{};
};

#endif

} // namespace rund::node::accel::detail
