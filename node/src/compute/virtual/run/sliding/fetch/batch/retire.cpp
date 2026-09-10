#include "../batch.hpp"

namespace rund::compute::detail::sliding_product_detail {

Status retire_fetch_batch(SlidingProductRun &state, FetchBatch &batch,
                          const Status status, const bool may_write) noexcept {
  Status result = status;
  auto sliding = state.pool->authority().sliding();
  for (std::size_t index = 0u; index < batch.count; ++index) {
    FetchBatchEntry &entry = batch.entries[index];
    if (entry.fetch.requires_backing()) {
      const std::uint64_t bytes = status ? entry.slice.bytes : 0u;
      if (!sliding.terminal_execution_sliding_fetch(
              state.sliding, entry.fetch, status,
              residency::execution::TerminalKind::Known, may_write, bytes)) {
        result = Status::fail(Reason::CompletionInvalid);
      }
    }
    if (!sliding.release_execution_sliding_fetch(
            state.sliding, std::move(entry.fetch))) {
      result = Status::fail(Reason::CompletionInvalid);
    }
  }
  return result;
}

} // namespace rund::compute::detail::sliding_product_detail
