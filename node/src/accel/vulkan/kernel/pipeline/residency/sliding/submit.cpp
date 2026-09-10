#include "../../../../adapter/error.hpp"
#include "../../../../buffer/access.hpp"

#include "../mode.hpp"
#include "internal.hpp"

#include <algorithm>
#include <array>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace {

[[nodiscard]] bool
valid_submission_request(const VulkanPipeline *const pipeline,
                         const KernelCompletion completion, void *const user,
                         const KernelTiming timing,
                         const PipelineSubmitMode mode) noexcept {
  return ValidVulkanPipeline(pipeline) && pipeline->residency != nullptr &&
         !vulkan_residency_detail::owns(pipeline->residency->mode) &&
         pipeline->residency->sliding.ready_for_submit &&
         !pipeline->residency->sliding.quarantined.load(
             std::memory_order_acquire) &&
         !pipeline->adapter->residency_quarantined.load(
             std::memory_order_acquire) &&
         completion != nullptr && user != nullptr &&
         timing != KernelTiming::Dispatch &&
         mode == PipelineSubmitMode::Residency;
}

void reset_submission_counters(VulkanPipeline &pipeline) noexcept {
  pipeline.submitted_dispatch_count = 0u;
  pipeline.submitted_control_count = 0u;
  pipeline.submitted_reset_count = 0u;
  pipeline.submitted_reset_bytes = 0u;
}

[[nodiscard]] rund::AccelCheck
build_gate_commands(VulkanPipeline &pipeline, VulkanResidencySlidingGate &gate,
                    const BackendResidencySlidingDescriptor &descriptor,
                    const std::span<const std::uint32_t> locals,
                    VulkanResidencySlidingSubmissionStorage &storage,
                    std::size_t &command_count) noexcept {
  auto &commands = storage.commands;
  std::size_t base_count = 0u;
  const bool payload_ready = vulkan_sliding_detail::build_payload(
      pipeline, gate, descriptor, locals, storage.payload);
  const rund::AccelCheck built =
      payload_ready
          ? BuildVulkanResidencySubmission(
                pipeline, locals, commands, base_count,
                pipeline.submitted_dispatch_count,
                pipeline.submitted_control_count,
                pipeline.submitted_reset_count, pipeline.submitted_reset_bytes)
          : rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
  if (!built.ok) {
    return built;
  }
  if (base_count < 2u || base_count + 1u > commands.size() ||
      !UploadVulkanBuffer(gate.descriptor, &storage.payload,
                          sizeof(storage.payload))) {
    return {false, "accel_vulkan_memory_unavailable"};
  }
  std::move_backward(commands.begin() + 1u, commands.begin() + base_count,
                     commands.begin() + base_count + 1u);
  commands[1u] = gate.command.buffer;
  command_count = base_count + 1u;
  return {true, "ok"};
}

[[nodiscard]] bool
submit_gate_commands(VulkanPipeline &pipeline, VulkanResidencySlidingGate &gate,
                     const BackendResidencySlidingDescriptor &descriptor,
                     const std::shared_ptr<void> &prepared,
                     const VulkanResidencySlidingSubmissionStorage &storage,
                     const std::size_t command_count,
                     submission::State<VulkanPipeline> &state,
                     const KernelTiming timing, const char *&failure) noexcept {
  gate.active_owner = prepared;
  const bool submitted = vulkan_sliding_detail::queue_gate_submission(
      *pipeline.adapter, gate,
      std::span<const VkCommandBuffer>{storage.commands.data(), command_count},
      pipeline.residency->prefix.fence, vulkan_sliding_detail::complete, &state,
      timing == KernelTiming::Submission);
  if (submitted) {
    vulkan_sliding_detail::commit_descriptor(gate, descriptor);
  } else {
    gate.active_owner.reset();
    failure = VulkanLastError(pipeline.adapter);
  }
  return submitted;
}

} // namespace

rund::AccelCheck SubmitVulkanResidencySliding(
    const std::shared_ptr<void> &prepared,
    const BackendResidencySlidingDescriptor &descriptor,
    const KernelCompletion completion, void *const user,
    const KernelTiming timing, const PipelineSubmitMode mode,
    const std::span<const std::uint32_t> locals) noexcept {
  auto *const pipeline = static_cast<VulkanPipeline *>(prepared.get());
  if (!valid_submission_request(pipeline, completion, user, timing, mode)) {
    return {false, "accel_vulkan_pipeline_selection_unavailable"};
  }
  submission::State<VulkanPipeline> &state = pipeline->submission;
  if (!submission::Begin(state, *pipeline, completion, user)) {
    return {false, "compute_pipeline_busy"};
  }
  bool submitted = false;
  const char *failure = "accel_vulkan_pipeline_selection_unavailable";
  VulkanResidencySlidingSubmissionStorage storage{};
  std::size_t command_count = 0u;
  VulkanResidencySlidingGate &gate = pipeline->residency->sliding;
  vulkan_sliding_detail::observe_submit_frontier(gate);
  {
    std::lock_guard lock{pipeline->adapter->mutex};
    if (gate.quarantined.load(std::memory_order_acquire) ||
        pipeline->adapter->residency_quarantined.load(
            std::memory_order_acquire)) {
      failure = "compute_device_lost";
    } else {
      reset_submission_counters(*pipeline);
      const rund::AccelCheck built = build_gate_commands(
          *pipeline, gate, descriptor, locals, storage, command_count);
      if (!built.ok) {
        failure = built.reason;
      } else {
        submitted =
            submit_gate_commands(*pipeline, gate, descriptor, prepared, storage,
                                 command_count, state, timing, failure);
      }
    }
  }
  if (!submitted) {
    reset_submission_counters(*pipeline);
    submission::Cancel(state);
    return {false, failure};
  }
  return {true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
