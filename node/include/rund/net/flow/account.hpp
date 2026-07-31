#pragma once

#include <rund/counter.hpp>
#include <rund/net/flow/limit.hpp>
#include <rund/net/flow/result.hpp>

#include <cstdint>

namespace rund::net::flow {

namespace detail {

[[nodiscard]] constexpr Result reject(const State state,
                                      const std::uint64_t bytes,
                                      const ::rund::ReasonCode code) noexcept {
  State rejected_state = state;
  rejected_state.rejected_bytes =
      ::rund::detail::counter::SaturatingAdd(rejected_state.rejected_bytes,
                                             bytes);
  Result result{code};
  result.state = rejected_state;
  result.requested_bytes = bytes;
  return result;
}

[[nodiscard]] constexpr Result accept(const State state,
                                      const std::uint64_t bytes) noexcept {
  Result result{::rund::ReasonCode::Ok};
  result.state = state;
  result.requested_bytes = bytes;
  return result;
}

} // namespace detail

[[nodiscard]] constexpr Result reserve(const State state, const Limit limit,
                                       const std::uint64_t bytes) noexcept {
  if (bytes == 0u) {
    return detail::reject(state, bytes, ::rund::ReasonCode::NetFlowZeroBytes);
  }
  if (state.inflight_bytes > limit.max_inflight_bytes ||
      bytes > limit.max_inflight_bytes - state.inflight_bytes) {
    return detail::reject(state, bytes,
                          ::rund::ReasonCode::NetFlowInflightExceeded);
  }
  if (state.total_bytes > limit.max_total_bytes ||
      bytes > limit.max_total_bytes - state.total_bytes) {
    return detail::reject(state, bytes,
                          ::rund::ReasonCode::NetFlowTotalExceeded);
  }
  return detail::accept(State{.inflight_bytes = state.inflight_bytes + bytes,
                              .total_bytes = state.total_bytes + bytes,
                              .rejected_bytes = state.rejected_bytes},
                        bytes);
}

[[nodiscard]] constexpr Result release(const State state,
                                       const std::uint64_t bytes) noexcept {
  if (bytes == 0u) {
    return detail::reject(state, bytes, ::rund::ReasonCode::NetFlowZeroBytes);
  }
  if (bytes > state.inflight_bytes) {
    return detail::reject(state, bytes,
                          ::rund::ReasonCode::NetFlowReleaseExceeded);
  }
  return detail::accept(State{.inflight_bytes = state.inflight_bytes - bytes,
                              .total_bytes = state.total_bytes,
                              .rejected_bytes = state.rejected_bytes},
                        bytes);
}

[[nodiscard]] constexpr Result record_send(const State state, const Limit limit,
                                           const std::uint64_t bytes) noexcept {
  return reserve(state, limit, bytes);
}

[[nodiscard]] constexpr Result
record_receive(const State state, const Limit limit,
               const std::uint64_t bytes) noexcept {
  return reserve(state, limit, bytes);
}

} // namespace rund::net::flow
