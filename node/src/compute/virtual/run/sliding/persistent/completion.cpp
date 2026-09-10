#include "../internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

namespace {

[[nodiscard]] bool complete_closed_failure(
    SlidingProductRun &state, const bool closed, const Status expected,
    const residency::execution::SlidingEvidence *const evidence =
        nullptr) noexcept {
  const Status pipeline = finish_pipeline_sliding(state, expected);
  if (!closed || expected || pipeline ||
      pipeline.reason() != expected.reason()) {
    return false;
  }
  if (evidence != nullptr) {
    fold_service_stats(state, *evidence);
  }
  complete_known_final(state, pipeline);
  return true;
}

} // namespace

void complete_persistent_product(SlidingProductRun &state,
                                 Status service) noexcept {
  node::accel::detail::PersistentResidencySlidingFinal native{};
  bool final_received = false;
  {
    std::lock_guard lock{state.gate};
    if (!state.persistent_final_received) {
      service = Status::fail(Reason::CompletionInvalid);
      state.poison.store(true, std::memory_order_release);
    } else {
      native = state.persistent_final;
      final_received = true;
    }
  }
  if (final_received) {
    fold_native_stats(state, native);
  }
  Status result = service;
  if (final_received) {
    const Status backend = status_from(native.check);
    if (!backend || result) {
      result = backend;
    }
  }
  result = merged_run_status(state, result);
  const bool unknown =
      native.terminal == node::accel::detail::NativeTerminal::UnknownMayWrite ||
      state.poison.load(std::memory_order_acquire);
  if (unknown) {
    static_cast<void>(
        finish_pipeline_sliding(state, Status::fail(Reason::DeviceLost)));
    quarantine_final(state);
    return;
  }

  if (result) {
    PersistentPublication publication{};
    const Status prepared =
        prepare_persistent_publication(state, native, publication);
    if (!prepared) {
      if (publication.close == CloseRequirement::Required) {
        GenerationClose joined = close_generation(state, prepared);
        if (complete_closed_failure(state, joined.closed, joined.status,
                                    &joined.evidence)) {
          return;
        }
        state.poison.store(true, std::memory_order_release);
        quarantine_final(state);
        return;
      }
      if (publication.abort == residency::FinalAbort::Closed) {
        if (complete_closed_failure(state, true, prepared)) {
          return;
        }
        state.poison.store(true, std::memory_order_release);
        quarantine_final(state);
        return;
      }
      if (publication.abort == residency::FinalAbort::Quarantined ||
          publication.abort == residency::FinalAbort::Invalid) {
        static_cast<void>(
            finish_pipeline_sliding(state, Status::fail(Reason::DeviceLost)));
        quarantine_final(state);
        return;
      }
      static_cast<void>(
          finish_pipeline_sliding(state, Status::fail(Reason::DeviceLost)));
      quarantine_final(state);
      return;
    }
    complete_known_final(state, Status::success());
    return;
  }

  GenerationClose joined = close_generation(state, result);
  const Reason close_reason = joined.status.reason();
  const Status pipeline = finish_pipeline_sliding(state, joined.status);
  joined.status = pipeline;
  if (!pipeline && pipeline.reason() != close_reason) {
    state.poison.store(true, std::memory_order_release);
  }
  if (!joined.closed) {
    quarantine_final(state);
    return;
  }
  fold_service_stats(state, joined.evidence);
  complete_known_final(state, joined.status);
}

} // namespace rund::compute::detail::sliding_product_detail
