#include "../../../buffer/access.hpp"

#include "local.hpp"
#include "mode.hpp"
#include "plan.hpp"

#include "generated_indirect/internal.hpp"
#include "../../../command/resources.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] rund::AccelCheck
MintResidencyPreparationGeneration(VulkanPipeline &pipeline) noexcept {
  if (pipeline.adapter == nullptr) {
    return {false, "accel_vulkan_residency_generation_owner_unavailable"};
  }
  if (pipeline.preparation_generation != 0u) {
    return {false, "accel_vulkan_residency_generation_reentry"};
  }
  std::uint64_t expected =
      pipeline.adapter->residency_preparation_generation.load(
          std::memory_order_acquire);
  for (;;) {
    if (expected == std::numeric_limits<std::uint64_t>::max()) {
      return {false, "accel_vulkan_residency_generation_exhausted"};
    }
    const std::uint64_t next = expected + 1u;
    if (pipeline.adapter->residency_preparation_generation
            .compare_exchange_weak(expected, next, std::memory_order_acq_rel,
                                   std::memory_order_acquire)) {
      pipeline.preparation_generation = next;
      return {true, "ok"};
    }
  }
}

void ResetResidencyAdmissionSnapshot(VulkanPipeline &pipeline) noexcept {
  auto &snapshot = pipeline.residency_admission;
  snapshot = {};
  snapshot.candidate_count = 5u;
  snapshot.generation_tag = pipeline.preparation_generation;
  const std::array<VulkanResidencyAdmissionCandidate, 5u> candidates{
      VulkanResidencyAdmissionCandidate::GraphDirect,
      VulkanResidencyAdmissionCandidate::GraphStageGeneratedIndirect,
      VulkanResidencyAdmissionCandidate::GraphStageSequence,
      VulkanResidencyAdmissionCandidate::GeneratedIndirect,
      VulkanResidencyAdmissionCandidate::Direct};
  for (std::size_t index = 0u; index < candidates.size(); ++index) {
    snapshot.candidates[index].candidate = candidates[index];
    snapshot.candidates[index].generation_tag = snapshot.generation_tag;
  }
}

namespace {

void DestroySelection(VulkanResidencySelection &selection) noexcept {
  if (selection.active_persistent.load(std::memory_order_acquire) != nullptr) {
    return;
  }
  if (selection.adapter != nullptr) {
    if (vulkan_residency_detail::owns(selection.mode)) {
      vulkan_generated_indirect_detail::destroy(selection);
    } else {
      DestroyVulkanResidencySlidingGate(selection);
    }
    DestroyVulkanBuffer(*selection.adapter, selection.arguments);
    DestroyCommand(selection.adapter->device, selection.suffix);
    for (VulkanResidencyStep &step : selection.steps) {
      DestroyCommand(selection.adapter->device, step.command);
    }
    DestroyCommand(selection.adapter->device, selection.prefix);
  }
  selection.steps.clear();
  selection.original_arguments.clear();
  selection.argument_owners.clear();
  selection.active_window.store(nullptr, std::memory_order_release);
  selection.adapter = nullptr;
  selection.host_bytes = 0u;
  selection.mode =
      vulkan_residency_detail::mode(VulkanResidencyAdmissionCandidate::Direct);
  selection.ready = false;
}

} // namespace

VulkanResidencySelection::~VulkanResidencySelection() {
  DestroySelection(*this);
}

#endif

} // namespace rund::node::accel::detail
