#include "../../local.hpp"
#include "../local.hpp"

#include <cstdint>

namespace rund::node::test_contract::window {

[[nodiscard]] bool RuntimeShape(const rund::compute::Stats &stats,
                                const rund::compute::Backend backend,
                                const std::uint32_t count) noexcept {
  using rund::compute::PipelineStats;
  const std::uint64_t active = CeilDiv(count, kTile);
  const std::uint64_t submits =
      backend == rund::compute::Backend::Cpu ? 0u : 1u;
  const bool physical_shape =
      backend != rund::compute::Backend::Metal ||
      (stats.dispatches == 2u && stats.pipeline.control_command_count == 1u);
  const bool bounded_telemetry =
      backend == rund::compute::Backend::Cpu
          ? stats.control.generated_item_count == 0u &&
                stats.control.generated_capacity == 0u &&
                stats.control.indirect_dispatch_count == 0u &&
                stats.control.indirect_work_item_count == 0u
          : stats.control.generated_item_count == 2u * count &&
                stats.control.generated_capacity == 2u * active * kTile &&
                stats.control.indirect_dispatch_count == 2u * active &&
                stats.control.indirect_work_item_count == 2u * count;
  return physical_shape && bounded_telemetry &&
         stats.control.overflow_ordinal ==
             rund::compute::ControlStats::no_overflow &&
         stats.pipeline.status_entry_count ==
             (backend == rund::compute::Backend::Cpu ? 0u : 3u * kOuter) &&
         stats.command_submits == submits && stats.pipeline.step_count == 1u &&
         stats.pipeline.verified_step_count == 1u &&
         stats.pipeline.failed_step_index == PipelineStats::no_failed_step &&
         stats.pipeline.executed_outer_window_count == active &&
         stats.pipeline.skipped_outer_window_count == kOuter - active &&
         stats.pipeline.executed_inner_iteration_count == active * kInner &&
         stats.pipeline.skipped_inner_iteration_count ==
             (kOuter - active) * kInner &&
         stats.pipeline.failed_outer_window == PipelineStats::no_coordinate &&
         stats.pipeline.failed_inner_iteration ==
             PipelineStats::no_coordinate &&
         stats.pipeline.barrier_count == kTemplates - 1u &&
         stats.pipeline.prepared_template_count == kPreparedTemplates &&
         stats.pipeline.prepared_command_count == kCommands &&
         stats.control.iteration_count == active &&
         stats.control.skipped_iteration_count == kOuter - active &&
         WarmSetupClean(stats);
}

} // namespace rund::node::test_contract::window
