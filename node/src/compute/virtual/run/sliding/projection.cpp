#include "internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

bool valid_project_request(const SlidingProductRun *const state,
                           const std::uint64_t coordinate,
                           const std::uint64_t turn,
                           const std::uint8_t slot) noexcept {
  return state != nullptr && state->cold != nullptr && state->pool != nullptr &&
         state->input != nullptr && state->run != nullptr &&
         slot < state->cold->role_count &&
         coordinate < state->cold->plan.epoch_count() &&
         coordinate % state->cold->role_count == slot &&
         coordinate / state->cold->role_count == turn;
}

bool run_accepts_projection(SlidingProductRun &state) noexcept {
  std::lock_guard lock{state.gate};
  return static_cast<bool>(state.failure);
}

bool ensure_projection(SlidingProductRun &state, SlidingProductWork &work,
                       const std::uint64_t coordinate) noexcept {
  if (work.projected && work.projection.coordinate.ordinal == coordinate) {
    return true;
  }
  if (!reset_work(work, coordinate) ||
      !state.sliding.project(coordinate, work.uses, work.projection) ||
      work.projection.coordinate.ordinal != coordinate) {
    return false;
  }
  work.projected = true;
  work.output_count = work.projection.persist_count;
  return true;
}

} // namespace rund::compute::detail::sliding_product_detail
