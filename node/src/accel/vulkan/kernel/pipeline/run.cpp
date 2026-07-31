#include "state.hpp"

#include "../../../kernel/reset/stats.hpp"
#include "../../../kernel/telemetry.hpp"
#include "../../command.hpp"
#include "telemetry.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <chrono>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

void CompleteVulkanPipeline(void *const raw, KernelResult result) noexcept {
  auto *const state = static_cast<submission::State<VulkanPipeline> *>(raw);
  if (state == nullptr) {
    return;
  }
  const submission::Claim<VulkanPipeline> claim = submission::Take(*state);
  if (!claim) {
    return;
  }
  VulkanPipeline *const pipeline = claim.owner;
  if (pipeline->dispatch_count != 0u) {
    std::lock_guard lock{pipeline->adapter->mutex};
    result.pipeline.submitted = true;
    result.pipeline.control_command_count = pipeline->control.command_count;
    if (result.check.ok) {
      const auto control_start = std::chrono::steady_clock::now();
      result.pipeline.control_observed =
          ReadVulkanPipelineControl(pipeline->control, result.pipeline.control);
      const auto control_end = std::chrono::steady_clock::now();
      result.pipeline.control_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(control_end -
                                                               control_start)
              .count());
      if (!result.pipeline.control_observed) {
        result.check =
            rund::AccelCheck{false, "accel_vulkan_memory_unavailable"};
      } else {
        ProjectTelemetry(result.pipeline.control, result.stats);
        ObserveVulkanProfile(*pipeline, result);
        ::rund::detail::counter::Accumulate(pipeline->adapter->dispatch_count,
                                            pipeline->dispatch_count);
      }
    }
  }
  result.stats.dispatch_count = result.check.ok ? pipeline->dispatch_count : 0u;
  SetResetStats(result.stats, result.check.ok, pipeline->reset_count,
                pipeline->reset_bytes);
  result.stats.command_submit_count =
      pipeline->dispatch_count == 0u ? 0u : result.stats.command_submit_count;
  result.stats.command_capacity = pipeline->dispatch_count == 0u ? 0u : 1u;
  result.stats.command_inflight_peak = pipeline->dispatch_count == 0u ? 0u : 1u;
  claim.completion(claim.user, result);
}

[[nodiscard]] KernelResult
VulkanPipelineResult(VulkanPipeline *const pipeline = nullptr) noexcept {
  KernelResult result{
      .check = rund::AccelCheck{true, "ok"},
      .stats = rund::RuntimeStats{.ok = true, .reason = "ok"},
  };
  if (pipeline != nullptr && pipeline->profile != nullptr) {
    VulkanPipelineProfile &profile = *pipeline->profile;
    for (std::size_t declared = 0u; declared < profile.declared_step_count;
         ++declared) {
      profile.rows[declared].work_sample_count = 1u;
    }
    result.pipeline.profile = PreparedPipelineProfileEvidence{
        .steps =
            std::span<const PreparedPipelineStepEvidence>{
                profile.rows.data(), profile.declared_step_count},
        .observed = true,
    };
  }
  return result;
}

rund::AccelCheck
SeedPreparedVulkanPipelineGeneration(const std::shared_ptr<void> &prepared,
                                     const std::uint32_t generation) noexcept {
  auto *const pipeline = static_cast<VulkanPipeline *>(prepared.get());
  if (!ValidVulkanPipeline(pipeline)) {
    return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
  }
  std::scoped_lock lock{pipeline->submission.mutex, pipeline->adapter->mutex};
  if (pipeline->submission.active()) {
    return rund::AccelCheck{false, "compute_pipeline_busy"};
  }
  if (pipeline->dispatch_count == 0u) {
    return rund::AccelCheck{true, "ok"};
  }
  const PreparedPipelineControl initial{.generation = generation};
  return UploadVulkanBuffer(pipeline->control.summary, &initial,
                            sizeof(initial))
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, "accel_vulkan_memory_unavailable"};
}

rund::AccelCheck
SubmitPreparedVulkanPipeline(const std::shared_ptr<void> &prepared,
                             const KernelCompletion completion_fn,
                             void *const user) noexcept {
  auto *const pipeline = static_cast<VulkanPipeline *>(prepared.get());
  if (!ValidVulkanPipeline(pipeline) || completion_fn == nullptr ||
      user == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  submission::State<VulkanPipeline> &state = pipeline->submission;
  if (!submission::Begin(state, *pipeline, completion_fn, user)) {
    return rund::AccelCheck{false, "compute_pipeline_busy"};
  }
  bool submitted = false;
  const char *failure_reason = "compute_pipeline_busy";
  {
    std::lock_guard lock{pipeline->adapter->mutex};
    if (pipeline->dispatch_count == 0u) {
      submitted = true;
    } else {
      const bool reset = ResetVulkanWindow(pipeline->window);
      submitted = reset && SubmitVulkanExternal(*pipeline->adapter,
                                                pipeline->command.buffer,
                                                pipeline->command.fence,
                                                CompleteVulkanPipeline, &state);
      if (!submitted) {
        failure_reason = reset ? VulkanLastError(pipeline->adapter)
                               : "accel_vulkan_memory_unavailable";
      }
    }
  }
  if (!submitted) {
    submission::Cancel(state);
    return rund::AccelCheck{false, failure_reason};
  }
  if (pipeline->dispatch_count == 0u) {
    CompleteVulkanPipeline(&state, VulkanPipelineResult(pipeline));
  }
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
