#include "../internal.hpp"

#include "../../../../../../kernel/prepared/interface/api.hpp"
#include "../../../../../../kernel/prepared/model.hpp"

#include <cstddef>

namespace rund::node::accel::detail::vulkan_generated_indirect_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

bool shape(const VulkanResidencySelection &selection, std::size_t &local_count,
           std::size_t &command_count) noexcept {
  local_count = 0u;
  command_count = 0u;
  if (!vulkan_residency_detail::owns(selection.mode)) {
    return false;
  }
  if (selection.mode == VulkanResidencyMode::GraphStageGeneratedIndirect) {
    local_count = selection.graph_generated.local_count;
    command_count = selection.graph_generated_command_count;
  } else {
    local_count = selection.graph_sequence.local_count;
    command_count = selection.graph_sequence_command_count;
  }
  return true;
}

bool frame(const VulkanResidencySelection &selection, const std::size_t index,
           bool &ready, bool &command, bool &quarantined, bool &gate) noexcept {
  ready = false;
  command = false;
  quarantined = false;
  gate = false;
  if (!vulkan_residency_detail::owns(selection.mode)) {
    return false;
  }
  if (selection.mode == VulkanResidencyMode::GraphStageGeneratedIndirect) {
    if (index >= selection.graph_generated_frames.size()) {
      return false;
    }
    const auto &row = selection.graph_generated_frames[index];
    ready = row.ready;
    command = row.command.buffer != VK_NULL_HANDLE;
    quarantined = row.quarantined;
    gate = row.gate.buffer != VK_NULL_HANDLE;
    return true;
  }
  if (index >= selection.graph_sequence_frames.size()) {
    return false;
  }
  const auto &row = selection.graph_sequence_frames[index];
  ready = row.command_ready;
  command = row.command.buffer != VK_NULL_HANDLE;
  quarantined = row.quarantined;
  gate = row.gates[0u].buffer != VK_NULL_HANDLE &&
         row.gates[1u].buffer != VK_NULL_HANDLE;
  return true;
}

Projection project(const VulkanPipeline &pipeline,
                   const VulkanResidencySelection &selection) noexcept {
  Projection result{};
  if (!vulkan_residency_detail::owns(selection.mode)) {
    return result;
  }
  result.state = ProjectionState::Invalid;
  result.reason = "accel_vulkan_residency_graph_shape_invalid";
  if (!shape(selection, result.local_count, result.command_count)) {
    return result;
  }
  result.shape_valid = true;
  if (selection.prefix.buffer == VK_NULL_HANDLE) {
    result.reason = "accel_vulkan_residency_prefix_missing";
    return result;
  }
  if (selection.suffix.buffer == VK_NULL_HANDLE) {
    result.reason = "accel_vulkan_residency_suffix_missing";
    return result;
  }
  if (selection.graph_generated_phase !=
      VulkanResidencyGraphGeneratedPhase::Recorded) {
    result.reason = "accel_vulkan_residency_graph_gate_not_ready";
    return result;
  }
  if (result.command_count == 0u) {
    result.reason = "accel_vulkan_residency_graph_commands_missing";
    return result;
  }
  if (result.command_count != result.local_count) {
    result.reason = "accel_vulkan_residency_graph_command_count_mismatch";
    return result;
  }
  for (std::size_t index = 0u; index < result.local_count; ++index) {
    bool frame_ready = false;
    bool frame_command = false;
    bool frame_quarantined = false;
    bool frame_gate = false;
    if (!frame(selection, index, frame_ready, frame_command, frame_quarantined,
               frame_gate)) {
      result.reason = "accel_vulkan_residency_graph_frame_invalid";
      return result;
    }
    if (!frame_ready) {
      result.reason = "accel_vulkan_residency_graph_frame_not_ready";
      return result;
    }
    if (!frame_command) {
      result.reason = "accel_vulkan_residency_graph_command_missing";
      return result;
    }
    if (!frame_gate) {
      result.reason = "accel_vulkan_residency_graph_gate_missing";
      return result;
    }
    if (frame_quarantined) {
      result.reason = "accel_vulkan_residency_graph_frame_quarantined";
      return result;
    }
  }
  if (pipeline.record == nullptr || !matches(pipeline, selection)) {
    result.reason = "accel_vulkan_residency_graph_proof_mismatch";
    return result;
  }
  result.state = ProjectionState::Ready;
  result.reason = "ok";
  return result;
}

#endif

} // namespace rund::node::accel::detail::vulkan_generated_indirect_detail

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] const VulkanPipeline *UnwrapPreparedVulkanPipeline(
    const PreparedKernelPipeline &prepared_pipeline) noexcept {
  if (!prepared_pipeline.ok || prepared_pipeline.owner == nullptr) {
    return nullptr;
  }
  const auto *const prepared_state =
      static_cast<const prepared::PipelineState *>(
          prepared_pipeline.owner.get());
  if (prepared_state == nullptr || prepared_state->ops == nullptr ||
      prepared_state->ops->api != rund::AccelApi::Vulkan ||
      prepared_state->backend == nullptr) {
    return nullptr;
  }
  const auto *const pipeline =
      static_cast<const VulkanPipeline *>(prepared_state->backend.get());
  return ValidVulkanPipeline(pipeline) ? pipeline : nullptr;
}

bool InspectVulkanResidencyAdmission(
    const PreparedKernelPipeline &prepared_pipeline,
    VulkanResidencyAdmissionSnapshot &snapshot) noexcept {
  snapshot = {};
  const auto *const pipeline = UnwrapPreparedVulkanPipeline(prepared_pipeline);
  if (pipeline == nullptr) {
    return false;
  }
  snapshot = pipeline->residency_admission;
  return true;
}

bool InspectVulkanResidencyStatus(
    const PreparedKernelPipeline &prepared_pipeline,
    VulkanResidencyStatus &status) noexcept {
  status = {};
  const auto *const pipeline = UnwrapPreparedVulkanPipeline(prepared_pipeline);
  if (pipeline == nullptr) {
    return false;
  }
  return EvaluateVulkanResidencyStatus(*pipeline, status);
}

bool InspectVulkanGraphGeneratedDiagnostics(
    const PreparedKernelPipeline &prepared_pipeline,
    VulkanResidencyGraphGeneratedDiagnostics &diagnostics) noexcept {
  diagnostics = {};
  const auto *const pipeline = UnwrapPreparedVulkanPipeline(prepared_pipeline);
  if (pipeline == nullptr || pipeline->residency == nullptr ||
      (pipeline->residency->mode !=
           VulkanResidencyMode::GraphStageGeneratedIndirect &&
       pipeline->residency->mode != VulkanResidencyMode::GraphStageSequence)) {
    return false;
  }
  diagnostics = pipeline->residency->graph_generated_diagnostics;
  diagnostics.phase = pipeline->residency->graph_generated_phase;
  diagnostics.generation = pipeline->residency->graph_generated_generation;
  diagnostics.submit_seq = pipeline->residency->submit_seq;
  diagnostics.done_seq = pipeline->residency->done_seq;
  diagnostics.submit_total = pipeline->residency->submit_total;
  diagnostics.done_total = pipeline->residency->done_total;
  diagnostics.accept_total = pipeline->residency->accept_total;
  diagnostics.known_total = pipeline->residency->known_total;
  diagnostics.unknown_total = pipeline->residency->unknown_total;
  return true;
}

#endif

} // namespace rund::node::accel::detail
