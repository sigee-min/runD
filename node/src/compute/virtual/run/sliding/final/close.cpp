#include "../../../../../compute/device/residency/execution/sliding/final/abort.hpp"
#include "../internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

GenerationClose close_generation(SlidingProductRun &state,
                                 Status result) noexcept {
  GenerationClose joined{.status = result};
  residency::execution::SlidingFinal frozen{};
  residency::ExecutionSlidingFinal authority_final{};
  residency::FinalAbort abort = residency::FinalAbort::Invalid;
  auto sliding = state.pool->authority().sliding();
  if (!state.sliding.prepare_final(frozen)) {
    const residency::FinalAbort closed =
        sliding.close_execution_sliding(
            state.cold->plan, state.sliding,
            result ? Status::fail(Reason::CompletionInvalid) : result);
    if (closed == residency::FinalAbort::Closed) {
      joined.status = Status::fail(Reason::CompletionInvalid);
      joined.closed = true;
      return joined;
    }
    joined.status = Status::fail(Reason::DeviceLost);
    state.poison.store(true, std::memory_order_release);
    return joined;
  }
  joined.evidence = frozen.evidence();
  if (!sliding.prepare_execution_sliding(
          state.cold->plan, state.sliding, frozen, authority_final, &abort)) {
    if (abort == residency::FinalAbort::Closed) {
      joined.status = Status::fail(Reason::CompletionInvalid);
      joined.closed = true;
      return joined;
    }
    if (abort == residency::FinalAbort::Quarantined) {
      joined.status = Status::fail(Reason::DeviceLost);
      state.poison.store(true, std::memory_order_release);
      return joined;
    }
    const residency::FinalAbort closed =
        sliding.close_execution_sliding(
            state.cold->plan, state.sliding,
            Status::fail(Reason::CompletionInvalid));
    if (closed == residency::FinalAbort::Closed) {
      joined.status = Status::fail(Reason::CompletionInvalid);
      joined.closed = true;
      return joined;
    }
    joined.status = Status::fail(Reason::DeviceLost);
    state.poison.store(true, std::memory_order_release);
    return joined;
  }

  const RebaseResult rebased = rebase_pipeline_sliding(state, joined.status);
  if (rebased.disposition == RebaseDisposition::Poisoned) {
    const residency::FinalAbort aborted =
        sliding.abort_execution_sliding(
            state.cold->plan, state.sliding, frozen, &authority_final,
            Status::fail(Reason::PipelinePoisoned));
    if (aborted != residency::FinalAbort::Quarantined) {
      // The authenticated poison terminal must retain the Authority lease;
      // do not let a run-only poison escape if the paired abort ever rejects.
      const residency::FinalAbort forced =
          sliding.close_execution_sliding(
              state.cold->plan, state.sliding,
              Status::fail(Reason::PipelinePoisoned));
      switch (forced) {
      case residency::FinalAbort::Quarantined:
        // The retained lease is now explicitly quarantined.
        break;
      case residency::FinalAbort::Closed:
        // Closed proves the token was cleared; publication remains failed.
        joined.closed = true;
        break;
      case residency::FinalAbort::Invalid:
        // Invalid is still a run-level poison and cannot publish success.
        break;
      }
    }
    state.poison.store(true, std::memory_order_release);
    joined.status = Status::fail(Reason::DeviceLost);
    return joined;
  }
  if (rebased.disposition == RebaseDisposition::RestoredFailure) {
    const residency::FinalAbort aborted =
        sliding.abort_execution_sliding(
            state.cold->plan, state.sliding, frozen, &authority_final,
            rebased.status);
    if (aborted == residency::FinalAbort::Closed) {
      joined.status = rebased.status;
      joined.closed = true;
      return joined;
    }
    joined.status = Status::fail(Reason::DeviceLost);
    state.poison.store(true, std::memory_order_release);
    return joined;
  }
  joined.status = rebased.status;
  const bool frozen_success = static_cast<bool>(frozen.evidence().status);
  const residency::ExecutionClose closed =
      sliding.accept_execution_sliding(
          state.cold->plan, state.sliding, frozen, std::move(authority_final),
          joined.evidence);
  if (closed.quarantined || !closed || closed.success != frozen_success) {
    joined.status = Status::fail(Reason::CompletionInvalid);
    state.poison.store(true, std::memory_order_release);
    return joined;
  }
  joined.closed = true;
  if (joined.status) {
    joined.status = joined.evidence.status;
  }
  return joined;
}

} // namespace rund::compute::detail::sliding_product_detail
