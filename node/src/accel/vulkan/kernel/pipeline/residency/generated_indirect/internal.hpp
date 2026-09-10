#pragma once

#include "submit_plan.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace rund::node::accel::detail::vulkan_generated_indirect_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool eligible(const VulkanPipeline &pipeline) noexcept;

struct GraphAdmission final {
  VulkanResidencyGraphStageGeneratedProof graph{};
  VulkanResidencyGraphStageSequenceProof sequence{};
  std::optional<VulkanResidencyAdmissionCandidate> candidate{};
};

[[nodiscard]] bool matches(const VulkanPipeline &pipeline,
                           const VulkanResidencySelection &selection) noexcept;

[[nodiscard]] GraphAdmission admit_graph(VulkanPipeline &pipeline) noexcept;

[[nodiscard]] rund::AccelCheck
prepare(VulkanPipeline &pipeline, VulkanResidencySelection &selection) noexcept;

[[nodiscard]] bool record(VulkanPipeline &pipeline,
                          VulkanResidencySelection &selection,
                          std::span<const std::uint32_t> locals) noexcept;

[[nodiscard]] bool finish(VulkanPipeline &pipeline,
                          KernelResult &result) noexcept;

void commit(VulkanPipeline &pipeline, const Plan &plan) noexcept;

void destroy(VulkanResidencySelection &selection) noexcept;

[[nodiscard]] bool shape(const VulkanResidencySelection &selection,
                         std::size_t &local_count,
                         std::size_t &command_count) noexcept;

[[nodiscard]] bool frame(const VulkanResidencySelection &selection,
                         std::size_t index, bool &ready, bool &command,
                         bool &quarantined, bool &gate) noexcept;

enum class ProjectionState : std::uint8_t { NotGenerated, Ready, Invalid };

struct Projection final {
  ProjectionState state{ProjectionState::NotGenerated};
  std::size_t local_count{};
  std::size_t command_count{};
  bool shape_valid{};
  const char *reason{"not_generated"};
};

[[nodiscard]] Projection
project(const VulkanPipeline &pipeline,
        const VulkanResidencySelection &selection) noexcept;

namespace impl {

[[nodiscard]] bool
eligible_graph(const VulkanPipeline &pipeline,
               VulkanResidencyGraphStageGeneratedProof &proof) noexcept;

[[nodiscard]] bool
eligible_sequence(const VulkanPipeline &pipeline,
                  VulkanResidencyGraphStageSequenceProof &proof) noexcept;

[[nodiscard]] bool
eligible_entry(const VulkanPipeline &pipeline, std::size_t index,
               VulkanResidencyGraphStageGeneratedProof &proof) noexcept;

[[nodiscard]] bool
same_semantics(const VulkanResidencyGraphStageGeneratedProof &left,
               const VulkanResidencyGraphStageGeneratedProof &right) noexcept;

[[nodiscard]] bool
same_frame(const VulkanResidencyGraphStageGeneratedProof &left,
           const VulkanResidencyGraphStageGeneratedProof &right) noexcept;

[[nodiscard]] bool
graph_proof_matches(const VulkanPipeline &pipeline,
                    const VulkanResidencySelection &selection) noexcept;

[[nodiscard]] bool
sequence_proof_matches(const VulkanPipeline &pipeline,
                       const VulkanResidencySelection &selection) noexcept;

[[nodiscard]] rund::AccelCheck
prepare_graph(VulkanPipeline &pipeline,
              VulkanResidencySelection &selection) noexcept;

[[nodiscard]] rund::AccelCheck
prepare_sequence(VulkanPipeline &pipeline,
                 VulkanResidencySelection &selection) noexcept;

[[nodiscard]] bool record_graph(VulkanPipeline &pipeline,
                                VulkanResidencySelection &selection,
                                std::span<const std::uint32_t> locals) noexcept;

[[nodiscard]] bool
record_sequence(VulkanPipeline &pipeline, VulkanResidencySelection &selection,
                std::span<const std::uint32_t> locals) noexcept;

void destroy_graph(VulkanResidencySelection &selection) noexcept;

void destroy_sequence(VulkanResidencySelection &selection) noexcept;

} // namespace impl

#endif

} // namespace rund::node::accel::detail::vulkan_generated_indirect_detail
