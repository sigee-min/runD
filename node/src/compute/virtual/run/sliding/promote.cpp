#include "internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

InputServiceResult service_promote(SlidingProductRun &state,
                                   SlidingProductWork &work,
                                   const std::uint64_t coordinate,
                                   Status &failure) noexcept {
  if (work.promoted) {
    return InputServiceResult::Ready;
  }
  auto sliding = state.pool->authority().sliding();
  residency::execution::SlidingPromote promote{};
  if (!sliding.issue_execution_sliding_promote(
          state.cold->plan, state.sliding, work.projection, work.uses,
          promote)) {
    return InputServiceResult::Pending;
  }
  Status promoted = copy_promote_frames(state, coordinate, promote);
  const std::uint64_t bytes = promoted ? promote.expected_bytes() : 0u;
  if (!sliding.terminal_execution_sliding_promote(
          state.sliding, promote, promoted,
          residency::execution::TerminalKind::Known,
          promote.transfer_mask() != 0u, bytes) ||
      !sliding.release_execution_sliding_promote(
          state.sliding, std::move(promote))) {
    promoted = Status::fail(Reason::CompletionInvalid);
  }
  if (!promoted) {
    failure = promoted;
    return InputServiceResult::Failed;
  }
  work.promoted = true;
  return InputServiceResult::Ready;
}

} // namespace rund::compute::detail::sliding_product_detail
