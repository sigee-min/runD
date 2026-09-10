#include "internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

OutputServiceResult service_drain(SlidingProductRun &state,
                                  SlidingProductWork &work,
                                  const std::size_t output,
                                  const std::size_t use,
                                  const BufferReadView source,
                                  const residency::FrameRegion device_region,
                                  Status &failure) noexcept {
  if (work.output_phase[output] != OutputServicePhase::NeedDrain) {
    return OutputServiceResult::Ready;
  }
  auto sliding = state.pool->authority().sliding();
  residency::execution::SlidingDrain drain{};
  if (!sliding.issue_execution_sliding_drain(
          state.cold->plan, state.sliding, work.projection, work.uses, use,
          drain)) {
    return OutputServiceResult::Pending;
  }
  Status drained = copy_drain_frame(state, source, device_region, drain);
  if (!sliding.terminal_execution_sliding_drain(
          state.sliding, drain, drained,
          residency::execution::TerminalKind::Known, true,
          drained ? drain.expected_bytes() : 0u) ||
      !sliding.release_execution_sliding_drain(
          state.sliding, std::move(drain))) {
    drained = Status::fail(Reason::CompletionInvalid);
  }
  if (!drained) {
    failure = drained;
    return OutputServiceResult::Failed;
  }
  work.output_phase[output] = OutputServicePhase::NeedPersist;
  return OutputServiceResult::Ready;
}

} // namespace rund::compute::detail::sliding_product_detail
