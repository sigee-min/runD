#pragma once

#include "../../../../kernel/residency/schedule.hpp"
#include <kernel/program/compute/graph/schema.hpp>

#include "model.hpp"

#include <accel/check.hpp>
#include <cstddef>
#include <cstdint>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanKernelResources;

struct ResidencyPlan final {
  VulkanResidencyAdmissionCandidate candidate{
      VulkanResidencyAdmissionCandidate::Direct};
  VulkanResidencyGraphStageDirectProof graph_direct{};
  VulkanResidencyGraphStageGeneratedProof graph_generated{};
  VulkanResidencyGraphStageSequenceProof graph_sequence{};
  bool bounded{};
};

[[nodiscard]] rund::AccelCheck
MintResidencyPreparationGeneration(VulkanPipeline &pipeline) noexcept;

void ResetResidencyAdmissionSnapshot(VulkanPipeline &pipeline) noexcept;

[[nodiscard]] bool SelectionEntry(const VulkanPipeline &pipeline,
                                  std::size_t index,
                                  VulkanKernelResources *&resources) noexcept;

[[nodiscard]] bool StepControlCount(const VulkanPipeline &pipeline,
                                    std::size_t index,
                                    std::uint64_t &count) noexcept;

[[nodiscard]] rund::AccelCheck
BuildResidencyPlan(VulkanPipeline &pipeline, ResidencyPlan &plan) noexcept;

#endif

} // namespace rund::node::accel::detail
