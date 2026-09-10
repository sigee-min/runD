#pragma once

#include "internal.hpp"

namespace rund::compute::detail::residency::physical::fetch {

template <typename Execution, typename State>
[[nodiscard]] bool
valid_credentials(const Execution &authority, const execution::Plan &plan,
                  const State &state,
                  const execution::SlidingProjection &projection,
                  const bool invocation_matches) noexcept {
  return authority.sliding_admitted && !authority.window_final &&
         authority.token != 0u && authority.native_inflight != 0u &&
         plan.identity() == authority.plan &&
         plan.epoch_count() == authority.epochs && !state.model_only &&
         state.authority_bound && !state.closed && !state.finalizing &&
         state.plan == Identity{.lo = authority.plan} &&
         state.token == authority.token &&
         state.generation == authority.generation &&
         state.owner == authority.native_inflight && invocation_matches &&
         state.accepts(projection.coordinate) &&
         projection.coordinate.ordinal >= state.admitted &&
         projection.coordinate.ordinal < state.planned;
}

} // namespace rund::compute::detail::residency::physical::fetch
