#include "state.hpp"

#include "../../../kernel/reset/stats.hpp"
#include "../../../kernel/telemetry.hpp"
#include "../../command.hpp"
#include "telemetry.hpp"
#include "trace.hpp"

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
    const bool trace_active = pipeline->trace_active;
    pipeline->trace_active = false;
    result.pipeline.submitted = true;
    result.pipeline.control_command_count = pipeline->control.command_count;
    if (result.check.ok && trace_active) {
      result.check = FoldVulkanPipelineDispatchTrace(*pipeline, result.stats);
    }
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
    if (pipeline->transfer.ready) {
      const bool transferred =
          result.check.ok && pipeline->transfer.input_staged;
      pipeline->transfer.output_ready = transferred;
      if (transferred) {
        ::rund::detail::counter::Accumulate(
            pipeline->adapter->host_to_device_bytes,
            pipeline->transfer.input_bytes);
        ::rund::detail::counter::Accumulate(
            pipeline->adapter->device_to_host_bytes,
            pipeline->transfer.output_bytes);
      } else {
        pipeline->transfer.input_staged = false;
      }
    }
  }
  result.stats.run.work.dispatch_count =
      result.check.ok ? pipeline->dispatch_count : 0u;
  SetResetStats(result.stats, result.check.ok, pipeline->reset_count,
                pipeline->reset_bytes);
  result.stats.run.work.command_submit_count =
      pipeline->dispatch_count == 0u
          ? 0u
          : result.stats.run.work.command_submit_count;
  result.stats.run.work.command_capacity =
      pipeline->dispatch_count == 0u ? 0u : 1u;
  result.stats.run.work.command_inflight_peak =
      pipeline->dispatch_count == 0u ? 0u : 1u;
  claim.completion(claim.user, result);
}

[[nodiscard]] KernelResult
VulkanPipelineResult(VulkanPipeline *const pipeline = nullptr) noexcept {
  KernelResult result{
      .check = rund::AccelCheck{true, "ok"},
      .stats = rund::RuntimeStats{.outcome = {.ok = true, .reason = "ok"}},
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
                             void *const user, const KernelTiming timing,
                             const PipelineSubmitMode) noexcept {
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
    if (pipeline->transfer.ready && !pipeline->transfer.input_staged) {
      failure_reason = "accel_vulkan_transfer_invalid";
    } else if (pipeline->transfer.ready && timing == KernelTiming::Dispatch) {
      failure_reason = "compute_telemetry_trace_unavailable";
    } else if (pipeline->dispatch_count == 0u) {
      submitted = true;
    } else if (timing == KernelTiming::Dispatch) {
      const rund::AccelCheck traced =
          EnsureVulkanPipelineDispatchTrace(*pipeline);
      rund::AccelCheck encoded = traced;
      bool command_open = false;
      if (encoded.ok) {
        command_open = EnsureVulkanCommandResources(*pipeline->adapter) &&
                       BeginVulkanCommand(*pipeline->adapter);
      }
      if (encoded.ok && !command_open) {
        encoded = rund::AccelCheck{false, VulkanLastError(pipeline->adapter)};
      }
      if (encoded.ok) {
        encoded = EncodeVulkanPipelineDispatchTrace(*pipeline);
      }
      if (!encoded.ok) {
        if (command_open) {
          CancelVulkanCommand(*pipeline->adapter);
        }
        failure_reason = encoded.reason;
      } else {
        pipeline->trace_active = true;
        submitted = SubmitVulkanCommand(*pipeline->adapter,
                                        CompleteVulkanPipeline, &state, false);
        if (!submitted) {
          pipeline->trace_active = false;
          failure_reason = VulkanLastError(pipeline->adapter);
        }
      }
    } else {
      submitted = SubmitVulkanExternal(
          *pipeline->adapter, pipeline->command.buffer, pipeline->command.fence,
          CompleteVulkanPipeline, &state, timing == KernelTiming::Submission);
      if (!submitted) {
        failure_reason = VulkanLastError(pipeline->adapter);
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
