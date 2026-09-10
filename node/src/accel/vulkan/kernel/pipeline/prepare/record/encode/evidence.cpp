#include "local.hpp"

#include "../../../telemetry.hpp"

namespace rund::node::accel::detail::vulkan_record_detail {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
rund::AccelCheck EncodeEvidence(const Encoding &context,
                                const StepEvidence &evidence,
                                std::size_t step_index) noexcept {
  auto &pipeline = context.pipeline;
  auto &recipe = context.recipe;
  const auto recording = context.recording;
  const PreparedProgramStatusSlice status_slice =
      recipe.status_steps[evidence.status_range.first + step_index];
  const std::size_t status_end =
      static_cast<std::size_t>(status_slice.first) + status_slice.count;
  if (status_end > recipe.canonical.size()) {
    return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
  }
  if (status_slice.count != 0u) {
    EncodeVulkanComputeToComputeBarrier(recording);
  }
  for (std::size_t status_index = status_slice.first; status_index < status_end;
       ++status_index) {
    const VulkanPipelineCanonicalStatus &current =
        recipe.canonical[status_index];
    if (current.active_program != evidence.template_index) {
      return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
    }
    if (!EncodeVulkanPipelineCanonicalStatus(recording, pipeline.control,
                                             current) ||
        !FoldVulkanPipelineControl(
            recording, pipeline.control,
            PreparedProgramStatusSlice{.first = current.first,
                                       .count = current.source.count},
            recipe.status.declared_steps[evidence.template_index],
            evidence.failed_outer_window, evidence.failed_inner_iteration,
            evidence.failed_nested_phase)) {
      return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
    }
  }
  const PreparedProgramStatusSlice telemetry_slice =
      recipe.telemetry_steps[evidence.telemetry_range.first + step_index];
  const std::size_t telemetry_end =
      static_cast<std::size_t>(telemetry_slice.first) + telemetry_slice.count;
  if (telemetry_end > pipeline.telemetry.size() ||
      !EncodeVulkanTelemetry(
          pipeline, recording,
          std::span<const VulkanPipelineTelemetryRecord>{pipeline.telemetry}
              .subspan(telemetry_slice.first, telemetry_slice.count),
          evidence.window_state, status_slice.count == 0u)) {
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }
  if (status_slice.count != 0u && telemetry_slice.count == 0u) {
    EncodeVulkanComputeToComputeBarrier(recording);
  }
  return rund::AccelCheck{true, ""};
}

#endif
} // namespace rund::node::accel::detail::vulkan_record_detail
