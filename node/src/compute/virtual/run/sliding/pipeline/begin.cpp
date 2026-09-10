#include "../internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

Status begin_pipeline_sliding(SlidingProductRun &state) noexcept {
  if (consume_persistent_pipeline_begin_failure_once()) {
    return Status::fail(Reason::PipelineBusy);
  }
  std::array<std::uint32_t, residency::execution::UseCapacity> locals{};
  const std::size_t count = state.run->frame_capacity;
  if (count == 0u || count > locals.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  for (std::size_t local = 0u; local < count; ++local) {
    locals[local] = static_cast<std::uint32_t>(local);
  }
  for (std::size_t bank = 0u; bank < state.cold->pipelines.size(); ++bank) {
    PipelineState &pipeline = *state.cold->pipelines[bank];
    Status started = Status::success();
    {
      std::lock_guard lock{pipeline.gate};
      started =
          start_pipeline(pipeline, PipelineClaimAuthority::PrivateResidency);
      if (started) {
        const PipelineOutcome prepared = prepare_residency_pipeline_execution(
            pipeline, std::span<const std::uint32_t>{locals.data(), count});
        started = prepared.status;
        if (!started && pipeline.phase == PipelinePhase::Running) {
          static_cast<void>(
              publish_residency_pipeline_execution(pipeline, count, prepared));
        }
      }
    }
    if (!started) {
      for (std::size_t prior = 0u; prior < bank; ++prior) {
        PipelineState &opened = *state.cold->pipelines[prior];
        std::lock_guard lock{opened.gate};
        if (state.pipeline_started[prior] &&
            opened.phase == PipelinePhase::Running) {
          PipelineOutcome rejected{
              .status = started,
              .publication_suppressed = true,
          };
          static_cast<void>(
              publish_residency_pipeline_execution(opened, count, rejected));
        }
        state.pipeline_started[prior] = false;
      }
      return started;
    }
    state.pipeline_started[bank] = true;
  }
  return Status::success();
}

} // namespace rund::compute::detail::sliding_product_detail
