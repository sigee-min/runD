#include "../../../../../compute/device/residency/execution/sliding/final/abort.hpp"
#include "../internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

bool close_rejected_start(SlidingProductRun &wait, residency::Pool &pool,
                          const SlidingProductOwner &owner,
                          const Status failure) noexcept {
  residency::execution::SlidingFinal frozen{};
  residency::ExecutionSlidingFinal authority_final{};
  residency::execution::SlidingEvidence evidence{};
  auto sliding = pool.authority().sliding();
  if (!sliding.reject_execution_sliding(owner.plan, wait.sliding, failure)) {
    const residency::FinalAbort closed =
        sliding.close_execution_sliding(owner.plan, wait.sliding, failure);
    if (closed == residency::FinalAbort::Closed) {
      return true;
    }
    quarantine_final(wait);
    return false;
  }
  if (!wait.sliding.prepare_final(frozen)) {
    const residency::FinalAbort closed =
        sliding.close_execution_sliding(owner.plan, wait.sliding, failure);
    if (closed == residency::FinalAbort::Closed) {
      return true;
    }
    quarantine_final(wait);
    return false;
  }
  residency::FinalAbort abort = residency::FinalAbort::Invalid;
  if (!sliding.prepare_execution_sliding(
          owner.plan, wait.sliding, frozen, authority_final, &abort)) {
    if (abort == residency::FinalAbort::Closed) {
      return true;
    }
    const residency::FinalAbort closed =
        sliding.close_execution_sliding(owner.plan, wait.sliding, failure);
    if (closed == residency::FinalAbort::Closed) {
      return true;
    }
    quarantine_final(wait);
    return false;
  }
  const residency::ExecutionClose closed =
      sliding.accept_execution_sliding(
          owner.plan, wait.sliding, frozen, std::move(authority_final),
          evidence);
  if (!closed || closed.quarantined) {
    quarantine_final(wait);
    return false;
  }
  return true;
}

} // namespace rund::compute::detail::sliding_product_detail
