#include "internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

PersistentServiceOutcome
make_persistent_unknown_failure(SlidingProductRun &state,
                                const Status failure) noexcept {
  state.poison.store(true, std::memory_order_release);
  return PersistentServiceOutcome{
      .status = failure,
      .disposition = PersistentServiceDisposition::UnknownFailure,
  };
}

PersistentServiceOutcome
make_persistent_known_failure(const Status failure) noexcept {
  return PersistentServiceOutcome{
      .status = failure,
      .disposition = PersistentServiceDisposition::KnownFailure,
  };
}

PersistentServiceOutcome make_persistent_completed() noexcept {
  return PersistentServiceOutcome{
      .status = Status::success(),
      .disposition = PersistentServiceDisposition::Completed,
  };
}

} // namespace rund::compute::detail::sliding_product_detail
