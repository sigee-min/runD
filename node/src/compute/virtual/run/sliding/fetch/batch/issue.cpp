#include "../batch.hpp"

namespace rund::compute::detail::sliding_product_detail {

FetchBatchIssue issue_fetch_batch(SlidingProductRun &state,
                                  SlidingProductWork &work,
                                  FetchBatch &batch) noexcept {
  if (work.next_fetch > work.projection.fetch_count ||
      work.projection.fetch_count - work.next_fetch > batch.entries.size()) {
    return FetchBatchIssue::PartialFailure;
  }
  const std::size_t remaining = work.projection.fetch_count - work.next_fetch;
  while (batch.count < remaining) {
    FetchBatchEntry &entry = batch.entries[batch.count];
    if (!state.pool->authority().sliding().issue_execution_sliding_fetch(
            state.cold->plan, state.sliding, work.projection, work.uses,
            work.next_fetch + batch.count, entry.fetch)) {
      return batch.count == 0u ? FetchBatchIssue::Pending
                               : FetchBatchIssue::PartialFailure;
    }
    ++batch.count;
    const residency::execution::FetchFill fill = entry.fetch.source().fill;
    if (fill == residency::execution::FetchFill::RepeatBoundary ||
        fill == residency::execution::FetchFill::ConstantBoundary) {
      break;
    }
  }
  return FetchBatchIssue::Ready;
}

} // namespace rund::compute::detail::sliding_product_detail
