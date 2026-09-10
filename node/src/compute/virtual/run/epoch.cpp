#include "epoch.hpp"

#include "epoch/internal.hpp"

#include "../stats.hpp"

#include <rund/counter.hpp>

namespace rund::compute::detail {
namespace {

[[nodiscard]] VirtualEpochResult
finish_epoch_failure(VirtualEpochContext &context,
                     const VirtualEpochPhaseResult &phase) noexcept {
  VirtualEpochResult result = abort_virtual_epoch(
      context, phase.status, phase.poison_pipeline, phase.invalidate_all);
  if (phase.failed_page != ResidencyStats::no_failed_page) {
    result.failed_page = phase.failed_page;
  }
  return result;
}

} // namespace

VirtualEpochResult execute_virtual_epoch(
    VirtualPipelineState &state, VirtualBacking &input, VirtualBacking &output,
    const VirtualRunProjection &run, const std::uint64_t epoch,
    std::array<bool, 2u> &prefetch_pending, Stats &stats,
    ::rund::node::hash_detail::Fnv &output_hash,
    VirtualReduction *const reduction, VirtualScan *const scan,
    VirtualRunTransaction *const transaction) noexcept {
  VirtualEpochContext context{
      .state = state,
      .input = input,
      .output = output,
      .run = run,
      .epoch = epoch,
      .prefetch_pending = prefetch_pending,
      .stats = stats,
      .output_hash = output_hash,
      .reduction = reduction,
      .scan = scan,
      .transaction = transaction,
  };
  const VirtualEpochResult admitted = admit_virtual_epoch(context);
  if (!admitted.status) {
    return admitted;
  }

  const VirtualEpochPhaseResult supplied = supply_virtual_epoch(context);
  if (!supplied.status) {
    return finish_epoch_failure(context, supplied);
  }

  Status status = execute_virtual_epoch_pipeline(context);
  if (!status) {
    const Stats execution_stats = pipeline_stats(*context.selected);
    const Status folded = accumulate_virtual_epoch(stats, execution_stats);
    return finish_epoch_failure(
        context, VirtualEpochPhaseResult{
                     .status = folded ? status : folded,
                     .poison_pipeline = poisoned_pipeline(*context.selected),
                     .invalidate_all = true,
                 });
  }

  // One successful epoch execution is physical evidence even if subsequent
  // scan adjustment, output transfer, or publication fails. Count it once,
  // before those phases, not once per internal scan pipeline execution.
  ::rund::detail::counter::Accumulate(stats.pipeline.residency.epoch_count, 1u);
  const VirtualEpochPhaseResult output_phase =
      finish_virtual_epoch_output(context);
  if (!output_phase.status) {
    return finish_epoch_failure(context, output_phase);
  }
  const VirtualEpochPhaseResult drain_phase =
      drain_virtual_epoch_output(context);
  if (!drain_phase.status) {
    return finish_epoch_failure(context, drain_phase);
  }
  return {};
}

} // namespace rund::compute::detail
