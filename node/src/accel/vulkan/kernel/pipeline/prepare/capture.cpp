#include "context.hpp"
#include "../../../command.hpp"
#include "../../../command/resources.hpp"
#include "../../../../kernel/backend/pipeline/failure.hpp"
namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck CaptureVulkanPipeline(VulkanPipelinePreparation &preparation) {
  auto &pipeline = preparation.pipeline;
  auto &canonical = preparation.canonical;
  auto &status_steps = preparation.status_steps;
  auto &telemetry_steps = preparation.telemetry_steps;
  auto &status_ranges = preparation.status_ranges;
  auto &telemetry_ranges = preparation.telemetry_ranges;
  auto &window_dispatches = preparation.window_dispatches;
  auto &window_gate_count = preparation.window_gate_count;
  const auto &templates = preparation.templates;
  const auto &entries = preparation.entries;
  const auto &barriers = preparation.barriers;
  const auto &transducers = preparation.transducers;
  auto &status = *preparation.status;
  auto &failure_context = *preparation.failure_context;
  const auto &recurrence = preparation.recurrence;
  const rund::AccelCheck command_ready = CreateCommand(
      pipeline->adapter->device, pipeline->adapter->compute_queue_family,
      pipeline->command, CommandKind::ReusablePrimary);
  if (!command_ready.ok) {
    return FailVulkanPipeline(pipeline, command_ready.reason);
  }
  if (pipeline->profile != nullptr) {
    VulkanPipelineProfile &profile = *pipeline->profile;
    // Two field-major fills plus their transfer-to-compute dependency.
    profile.instrumentation_command_count = 3u;
    profile.instrumentation_byte_count =
        static_cast<std::uint64_t>(profile.declared_step_count) *
        PreparedPipelineStepControlBytes;
  }
  pipeline->record = std::make_unique<VulkanPipelineRecordRecipe>();
  const rund::AccelCheck recipe_ready = MakeVulkanPipelineRecordRecipe(
      entries, barriers, transducers, canonical, status_steps, telemetry_steps,
      status_ranges, telemetry_ranges, status, templates.size(),
      window_dispatches, window_gate_count, recurrence.ready(),
      *pipeline->record);
  if (!recipe_ready.ok) {
    return FailVulkanPipeline(pipeline, recipe_ready.reason);
  }
  failure_context.stage(PreparedPipelineFailureStage::BackendCapture);
  const rund::AccelCheck recorded = RecordVulkanPipeline(
      *pipeline, pipeline->command, CommandKind::ReusablePrimary, false,
      &failure_context);
  if (!recorded.ok) {
    return FailVulkanPipeline(pipeline, recorded.reason);
  }

  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
