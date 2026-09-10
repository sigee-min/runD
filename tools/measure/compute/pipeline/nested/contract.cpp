#include "model.hpp"

namespace rund::measure::compute::nested_repeat {

bool SerialStepEvidence(const ::rund::compute::Backend backend,
                        const ::rund::compute::Stats &stats) {
  using ::rund::compute::PipelineStats;
  return stats.backend == backend && stats.command_submits == 1u &&
         stats.dispatches == 1u && stats.pipeline.step_count == 1u &&
         stats.pipeline.verified_step_count == 1u &&
         stats.pipeline.failed_step_index == PipelineStats::no_failed_step &&
         stats.pipeline.sealed_repetition_count == 1u &&
         stats.pipeline.coalesced_repetition_count == 0u;
}

bool NestedEvidence(const ::rund::compute::Backend backend,
                    const ::rund::compute::Stats &stats,
                    const ::rund::compute::PipelinePlan &plan) {
  using ::rund::compute::PipelineNestedPhase;
  using ::rund::compute::PipelineStats;
  const bool physical_shape =
      backend != ::rund::compute::Backend::Metal ||
      (stats.dispatches == MetalNestedPhysicalCommands &&
       stats.pipeline.control_command_count == 1u);
  return physical_shape && stats.backend == backend &&
         stats.command_submits == 1u && stats.dispatches != 0u &&
         stats.pipeline.step_count == 1u &&
         stats.pipeline.verified_step_count == 1u &&
         stats.pipeline.failed_step_index == PipelineStats::no_failed_step &&
         stats.pipeline.failed_outer_window == PipelineStats::no_coordinate &&
         stats.pipeline.failed_inner_iteration ==
             PipelineStats::no_coordinate &&
         stats.pipeline.failed_nested_phase == PipelineNestedPhase::None &&
         stats.pipeline.executed_outer_window_count == Outer &&
         stats.pipeline.skipped_outer_window_count == 0u &&
         stats.pipeline.executed_inner_iteration_count == SerialSubmits &&
         stats.pipeline.skipped_inner_iteration_count == 0u &&
         stats.pipeline.prepared_template_count == Templates &&
         stats.pipeline.prepared_command_count == Commands &&
         plan.outer_window_count == Outer && plan.tile_capacity == Tile &&
         plan.inner_iteration_count == Inner &&
         plan.prepared_template_count == Templates &&
         plan.prepared_command_count == Commands;
}

} // namespace rund::measure::compute::nested_repeat
