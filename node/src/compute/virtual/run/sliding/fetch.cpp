#include "internal.hpp"

#include "fetch/batch.hpp"

namespace rund::compute::detail::sliding_product_detail {

InputServiceResult service_fetches(SlidingProductRun &state,
                                   SlidingProductWork &work,
                                   Status &failure) noexcept {
  if (work.next_fetch == work.projection.fetch_count) {
    return InputServiceResult::Ready;
  }
  while (work.next_fetch < work.projection.fetch_count) {
    FetchBatch batch{};
    const FetchBatchIssue issued = issue_fetch_batch(state, work, batch);
    if (issued == FetchBatchIssue::Pending) {
      return InputServiceResult::Pending;
    }
    if (issued == FetchBatchIssue::PartialFailure) {
      failure = retire_fetch_batch(
          state, batch, Status::fail(Reason::CompletionInvalid), false);
      return InputServiceResult::Failed;
    }
    const FetchBatchIo io = read_fetch_batch(state, batch);
    Status supplied =
        retire_fetch_batch(state, batch, io.io.status, io.may_write);
    if (io.backing) {
      FetchIo telemetry = io.io;
      telemetry.status = supplied;
      record_fetch_io(state, telemetry);
    }
    if (!supplied) {
      failure = supplied;
      return InputServiceResult::Failed;
    }
    work.next_fetch += batch.count;
  }
  return InputServiceResult::Ready;
}

} // namespace rund::compute::detail::sliding_product_detail
