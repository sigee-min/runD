#include "internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

OutputServiceResult service_persist(SlidingProductRun &state,
                                    SlidingProductWork &work,
                                    const std::size_t output,
                                    const std::size_t use,
                                    Status &failure) noexcept {
  if (work.output_phase[output] == OutputServicePhase::Done) {
    return OutputServiceResult::Ready;
  }
  auto sliding = state.pool->authority().sliding();
  residency::execution::SlidingPersist persist{};
  if (!sliding.issue_execution_sliding_persist(
          state.cold->plan, state.sliding, work.projection, work.uses, use,
          persist)) {
    return OutputServiceResult::Pending;
  }
  require_persist_recovery(state);
  PersistIo io = perform_persist_io(state, persist);
  Status persisted = io.status;
  if (!sliding.terminal_execution_sliding_persist(
          state.sliding, persist, persisted,
          residency::execution::TerminalKind::Known, true, io.bytes) ||
      !sliding.release_execution_sliding_persist(
          state.sliding, std::move(persist))) {
    persisted = Status::fail(Reason::CompletionInvalid);
  }
  io.status = persisted;
  record_persist_io(state, io);
  if (!persisted) {
    failure = persisted;
    return OutputServiceResult::Failed;
  }
  work.output_phase[output] = OutputServicePhase::Done;
  return OutputServiceResult::Ready;
}

} // namespace rund::compute::detail::sliding_product_detail
