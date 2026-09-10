#include "local.hpp"
#include "mode.hpp"
#include "graph_direct.hpp"

#include "generated_indirect/internal.hpp"
#include "generated_indirect/map.hpp"

#include "../prepare/record.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck
StageVulkanPipelineResidency(const std::shared_ptr<void> &prepared,
                             std::shared_ptr<void> &candidate,
                             std::uint64_t &retained_bytes) noexcept {
  candidate.reset();
  retained_bytes = 0u;
  auto *const pipeline = static_cast<VulkanPipeline *>(prepared.get());
  if (!ValidVulkanPipeline(pipeline) || pipeline->residency == nullptr ||
      !pipeline->residency->ready) {
    return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
  }
  // Selection commands are already cold-owned and included in prepared Host
  // memory. Vulkan exposes no exact command-pool byte query, so this stage
  // adds no guessed native byte mirror.
  candidate = prepared;
  return rund::AccelCheck{true, "ok"};
}

bool EvaluateVulkanResidencyStatus(const VulkanPipeline &pipeline,
                                   VulkanResidencyStatus &status) noexcept {
  status = {};
  status.valid = ValidVulkanPipeline(&pipeline);
  if (!status.valid) {
    return false;
  }
  const VulkanResidencySelection *const selection = pipeline.residency.get();
  if (selection == nullptr) {
    status.readiness_reason = "accel_vulkan_residency_absent";
    return true;
  }
  status.present = true;
  status.mode = selection->mode;
  status.bounded = selection->bounded;
  status.quarantined = selection->quarantined.load(std::memory_order_acquire) ||
                       selection->window.quarantined;
  status.adapter_quarantined =
      pipeline.adapter->residency_quarantined.load(std::memory_order_acquire);
  const vulkan_generated_indirect_detail::Projection generated =
      vulkan_generated_indirect_detail::project(pipeline, *selection);
  std::size_t local_count = selection->steps.size();
  std::size_t generated_command_count = 0u;
  if (generated.state !=
      vulkan_generated_indirect_detail::ProjectionState::NotGenerated) {
    if (!generated.shape_valid) {
      status.readiness_reason = generated.reason;
      return true;
    }
    local_count = generated.local_count;
    generated_command_count = generated.command_count;
  }
  status.local_count = static_cast<std::uint32_t>(std::min<std::size_t>(
      local_count, std::numeric_limits<std::uint32_t>::max()));
  const std::size_t command_count =
      vulkan_residency_detail::owns(selection->mode) ? generated_command_count
                                                     : selection->steps.size();
  status.command_count = static_cast<std::uint32_t>(std::min<std::size_t>(
      command_count, std::numeric_limits<std::uint32_t>::max()));
  status.step_buffer_count = 0u;
  for (const VulkanResidencyStep &step : selection->steps) {
    if (step.command.buffer != VK_NULL_HANDLE) {
      ++status.step_buffer_count;
    }
  }
  status.prefix_buffer = selection->prefix.buffer != VK_NULL_HANDLE;
  status.prefix_fence = selection->prefix.fence != VK_NULL_HANDLE;
  status.suffix_buffer = selection->suffix.buffer != VK_NULL_HANDLE;
  status.arguments_buffer = selection->arguments.buffer != VK_NULL_HANDLE;
  status.arguments_mapped = selection->arguments.mapped != nullptr;

  if (!selection->ready) {
    status.readiness_reason = "accel_vulkan_residency_not_ready";
    return true;
  }
  if (!selection->bounded) {
    status.readiness_reason = "accel_vulkan_residency_unbounded";
    return true;
  }
  if (selection->adapter != pipeline.adapter) {
    status.readiness_reason = "accel_vulkan_residency_adapter_mismatch";
    return true;
  }
  if (selection->quarantined.load(std::memory_order_acquire)) {
    status.readiness_reason = "accel_vulkan_residency_quarantined";
    return true;
  }
  if (selection->window.quarantined) {
    status.readiness_reason = "accel_vulkan_residency_window_quarantined";
    return true;
  }
  if (pipeline.adapter->fault_host_read_once.load(std::memory_order_relaxed)) {
    status.readiness_reason = "accel_vulkan_residency_host_read_fault";
    return true;
  }
  if (pipeline.adapter->fault_host_write_once.load(std::memory_order_relaxed)) {
    status.readiness_reason = "accel_vulkan_residency_host_write_fault";
    return true;
  }
  const rund::AccelCheck timeline =
      VulkanTimelineCapability(pipeline.adapter->timeline);
  if (!timeline.ok) {
    status.readiness_reason = timeline.reason == nullptr
                                  ? "accel_vulkan_timeline_invalid"
                                  : timeline.reason;
    return true;
  }
  if (!status.prefix_fence) {
    status.readiness_reason = "accel_vulkan_residency_prefix_fence_missing";
    return true;
  }
  if (pipeline.record == nullptr) {
    status.readiness_reason = "accel_vulkan_residency_record_missing";
    return true;
  }
  if (selection->steps.size() != pipeline.record->entries.size()) {
    status.readiness_reason = "accel_vulkan_residency_step_count_mismatch";
    return true;
  }
  if (vulkan_residency_detail::owns(selection->mode)) {
    if (generated.state !=
        vulkan_generated_indirect_detail::ProjectionState::Ready) {
      status.readiness_reason = generated.reason;
      return true;
    }
  } else {
    if (!status.prefix_buffer) {
      status.readiness_reason = "accel_vulkan_residency_prefix_missing";
      return true;
    }
    if (!status.suffix_buffer) {
      status.readiness_reason = "accel_vulkan_residency_suffix_missing";
      return true;
    }
    if (!status.arguments_buffer) {
      status.readiness_reason = "accel_vulkan_residency_arguments_missing";
      return true;
    }
    if (!status.arguments_mapped) {
      status.readiness_reason = "accel_vulkan_residency_arguments_unmapped";
      return true;
    }
    if ((selection->arguments.usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ==
        0u) {
      status.readiness_reason = "accel_vulkan_residency_arguments_not_storage";
      return true;
    }
    if (selection->original_arguments.empty()) {
      status.readiness_reason = "accel_vulkan_residency_arguments_empty";
      return true;
    }
    if (selection->original_arguments.size() !=
        selection->argument_owners.size()) {
      status.readiness_reason =
          "accel_vulkan_residency_argument_owner_count_mismatch";
      return true;
    }
  }
  if (vulkan_residency_detail::is_graph_direct(selection->mode) &&
      !GraphDirectProofMatches(pipeline, *selection)) {
    status.readiness_reason =
        "accel_vulkan_residency_graph_direct_proof_mismatch";
    return true;
  }
  status.ready = true;
  status.readiness_reason = "ok";
  return true;
}

rund::AccelCheck
VulkanPipelineResidencyReady(const std::shared_ptr<void> &prepared,
                             bool &ready) noexcept {
  ready = false;
  const auto *const pipeline =
      static_cast<const VulkanPipeline *>(prepared.get());
  VulkanResidencyStatus status{};
  if (!EvaluateVulkanResidencyStatus(*pipeline, status)) {
    return rund::AccelCheck{false, status.readiness_reason};
  }
  ready = status.ready;
  return rund::AccelCheck{true, "ok"};
}

void CommitVulkanPipelineResidency(const std::shared_ptr<void> &prepared,
                                   std::shared_ptr<void> candidate) noexcept {
  auto *const pipeline = static_cast<VulkanPipeline *>(prepared.get());
  if (!ValidVulkanPipeline(pipeline) || candidate.get() != prepared.get() ||
      pipeline->residency == nullptr || !pipeline->residency->ready) {
    return;
  }
  candidate.reset();
}

rund::AccelCheck BuildVulkanResidencySubmission(
    const VulkanPipeline &pipeline, const std::span<const std::uint32_t> locals,
    const std::span<VkCommandBuffer> storage, std::size_t &command_count,
    std::uint64_t &dispatch_count, std::uint64_t &control_count,
    std::uint64_t &reset_count, std::uint64_t &reset_bytes) noexcept {
  command_count = 0u;
  dispatch_count = 0u;
  control_count = 0u;
  reset_count = 0u;
  reset_bytes = 0u;
  const VulkanResidencySelection *const selection = pipeline.residency.get();
  if (selection == nullptr) {
    return rund::AccelCheck{false,
                            "accel_vulkan_pipeline_selection_unavailable"};
  }
  const auto map = vulkan_generated_indirect_detail::map_access(*selection);
  if (!selection->ready || locals.empty() ||
      locals.size() > selection->steps.size() || map.selected ||
      !map.recognized || storage.size() < locals.size() + 2u ||
      selection->prefix.fence == VK_NULL_HANDLE) {
    return rund::AccelCheck{false,
                            "accel_vulkan_pipeline_selection_unavailable"};
  }
  if (selection->quarantined.load(std::memory_order_acquire)) {
    return rund::AccelCheck{false,
                            "accel_vulkan_pipeline_selection_unavailable"};
  }
  if (vulkan_residency_detail::is_graph_direct(selection->mode) &&
      !GraphDirectProofMatches(pipeline, *selection)) {
    return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
  }
  if (vulkan_residency_detail::owns(selection->mode)) {
    // Graph-stage owners use a separate plan/commit transaction. Generic
    // residency callers must not submit them without that commit owner.
    return {false, "accel_vulkan_pipeline_selection_unavailable"};
  }
  if (selection->prefix.buffer == VK_NULL_HANDLE ||
      selection->suffix.buffer == VK_NULL_HANDLE) {
    return rund::AccelCheck{false,
                            "accel_vulkan_pipeline_selection_unavailable"};
  }
  storage[command_count++] = selection->prefix.buffer;
  control_count = 1u;
  for (std::size_t index = 0u; index < locals.size(); ++index) {
    const std::uint32_t local = locals[index];
    if (local >= selection->steps.size()) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (locals[prior] == local) {
        return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
      }
    }
    const VulkanResidencyStep &step = selection->steps[local];
    if (step.declared_step != local || step.command.buffer == VK_NULL_HANDLE ||
        step.dispatch_count == 0u ||
        step.dispatch_count >
            std::numeric_limits<std::uint64_t>::max() - dispatch_count ||
        step.control_count >
            std::numeric_limits<std::uint64_t>::max() - control_count ||
        step.reset_count >
            std::numeric_limits<std::uint64_t>::max() - reset_count ||
        step.reset_bytes >
            std::numeric_limits<std::uint64_t>::max() - reset_bytes) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    storage[command_count++] = step.command.buffer;
    dispatch_count += step.dispatch_count;
    control_count += step.control_count;
    reset_count += step.reset_count;
    reset_bytes += step.reset_bytes;
  }
  storage[command_count++] = selection->suffix.buffer;
  if (control_count == std::numeric_limits<std::uint64_t>::max()) {
    return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
  }
  ++control_count;
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
