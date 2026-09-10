#include "service/internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

Status service_persistent_recurrence(SlidingProductRun &state) noexcept {
  const std::uint64_t total = state.cold->plan.epoch_count();
  std::uint64_t next = 0u;
  while (next < total) {
    std::uint64_t end = 0u;
    {
      std::lock_guard lock{state.persistent_control.gate};
      end = node::accel::detail::persistent_sliding_accepted_end(
          state.persistent_control);
    }
    if (end <= next) {
      const Status submitted = submit_persistent(state);
      if (!submitted) {
        return submitted;
      }
      std::lock_guard lock{state.persistent_control.gate};
      end = node::accel::detail::persistent_sliding_accepted_end(
          state.persistent_control);
    }
    if (end <= next || end > total) {
      return Status::fail(Reason::CompletionInvalid);
    }
    for (std::uint64_t coordinate = next; coordinate < end; ++coordinate) {
      const PersistentServiceOutcome serviced =
          service_persistent_coordinate(state, coordinate);
      if (serviced.disposition == PersistentServiceDisposition::KnownFailure) {
        const Status drained = service_persistent_known_suffix(
            state, coordinate + 1u, serviced.status);
        return drained ? serviced.status : drained;
      }
      if (serviced.disposition == PersistentServiceDisposition::UnknownFailure) {
        return serviced.status;
      }
    }
    next = end;
  }
  std::lock_guard lock{state.gate};
  return state.persistent_final_received
             ? status_from(state.persistent_final.check)
             : Status::fail(Reason::CompletionInvalid);
}

} // namespace rund::compute::detail::sliding_product_detail
