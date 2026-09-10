#include "internal.hpp"

#include <utility>

namespace rund::node::accel::detail::vulkan_sliding_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace {

[[nodiscard]] VulkanPipeline *
submission_owner(submission::State<VulkanPipeline> *const state) noexcept {
  if (state == nullptr) {
    return nullptr;
  }
  std::lock_guard lock{state->mutex};
  return state->owner;
}

[[nodiscard]] bool
read_acquired_result(VulkanResidencySlidingGate &gate) noexcept {
  if (gate.descriptor.mapped == nullptr ||
      gate.descriptor.bytes < sizeof(VulkanResidencySlidingPayload)) {
    return false;
  }
  gate.gpu_result_read_count.fetch_add(1u, std::memory_order_relaxed);
  const auto *const payload =
      static_cast<const VulkanResidencySlidingPayload *>(
          gate.descriptor.mapped);
  const std::uint64_t expected_generation =
      static_cast<std::uint64_t>(payload->words[DescriptorGenerationWord]) |
      (static_cast<std::uint64_t>(payload->words[DescriptorGenerationWord + 1u])
       << 32u);
  const std::uint64_t observed_generation =
      static_cast<std::uint64_t>(
          payload->words[VulkanResidencySlidingObservedGenerationWord]) |
      (static_cast<std::uint64_t>(
           payload->words[VulkanResidencySlidingObservedGenerationWord + 1u])
       << 32u);
  return payload->words[VulkanResidencySlidingAcceptedWord] == 1u &&
         payload->words[VulkanResidencySlidingReasonWord] == 0u &&
         expected_generation != 0u &&
         observed_generation == expected_generation;
}

void classify_result(KernelResult &result, const bool accepted) noexcept {
  if (!result.check.ok) {
    result.terminal = NativeTerminal::UnknownMayWrite;
  } else if (!accepted) {
    result.check = {false, "accel_kernel_pipeline_invalid"};
    result.stats.outcome = {.ok = false, .reason = result.check.reason};
    result.terminal = NativeTerminal::UnknownMayWrite;
  }
}

void publish_unknown(VulkanPipeline &pipeline,
                     VulkanResidencySlidingGate &gate) noexcept {
  pause_terminal_frontier(gate);
  gate.quarantined.store(true, std::memory_order_release);
  pipeline.adapter->residency_quarantined.store(true,
                                                std::memory_order_release);
}

} // namespace

void complete(void *const raw, KernelResult result) noexcept {
  auto *const state = static_cast<submission::State<VulkanPipeline> *>(raw);
  VulkanPipeline *const pipeline = submission_owner(state);
  std::shared_ptr<void> retained{};
  if (pipeline != nullptr && pipeline->adapter != nullptr &&
      pipeline->residency != nullptr) {
    std::unique_lock frontier{pipeline->adapter->mutex};
    VulkanResidencySlidingGate &gate = pipeline->residency->sliding;
    retained = std::move(gate.active_owner);
    const bool acquired =
        result.check.ok && result.terminal == NativeTerminal::Known;
    const bool accepted = acquired && read_acquired_result(gate);
    classify_result(result, accepted);
    if (result.terminal == NativeTerminal::UnknownMayWrite) {
      publish_unknown(*pipeline, gate);
    }
    frontier.unlock();
  } else {
    classify_result(result, false);
  }
  CompleteVulkanPipeline(raw, std::move(result));
  retained.reset();
}

#endif

} // namespace rund::node::accel::detail::vulkan_sliding_detail
