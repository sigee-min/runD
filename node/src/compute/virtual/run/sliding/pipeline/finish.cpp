#include "../internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

Status finish_pipeline_sliding(SlidingProductRun &state,
                               const Status result) noexcept {
  Status folded = result;
  const std::size_t issued = state.run->frame_capacity;
  for (std::size_t bank = 0u; bank < state.cold->pipelines.size(); ++bank) {
    PipelineState &pipeline = *state.cold->pipelines[bank];
    std::lock_guard lock{pipeline.gate};
    if (!state.pipeline_started[bank] ||
        pipeline.phase != PipelinePhase::Running) {
      folded = Status::fail(Reason::CompletionInvalid);
      continue;
    }
    PipelineOutcome outcome{
        .status = folded,
        .verified = folded ? issued : 0u,
        .writes_possible = pipeline.attempt.writes_possible,
        .native_completion = state.pipeline_submitted[bank]
                                 ? PipelineNativeCompletion::Known
                                 : PipelineNativeCompletion::NotSubmitted,
        .publication_suppressed = !folded,
    };
    const Status published =
        publish_residency_pipeline_execution(pipeline, issued, outcome);
    state.pipeline_started[bank] = false;
    if (!published && (folded || published.reason() != folded.reason())) {
      folded = published;
    }
  }
  return folded;
}

} // namespace rund::compute::detail::sliding_product_detail
