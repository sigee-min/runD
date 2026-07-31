#include "test/assert.hpp"

#include "local.hpp"

#include <rund/net/flow.hpp>

#include <cstdint>
#include <limits>
#include <string_view>

int RunNetFlowRejectCase() {
  const rund::net::flow::Limit limit{.max_inflight_bytes = 10u,
                                     .max_total_bytes = 20u};
  const rund::net::flow::Result zero_bytes =
      rund::net::flow::reserve(rund::net::flow::State{}, limit, 0u);
  TEST_ASSERT(!zero_bytes.ok());
  TEST_ASSERT(zero_bytes.code() == rund::ReasonCode::NetFlowZeroBytes);
  TEST_ASSERT(std::string_view{zero_bytes.error()} == "net_flow_zero_bytes");
  TEST_ASSERT(zero_bytes.state.rejected_bytes == 0u);

  const rund::net::flow::Limit zero_limit{.max_inflight_bytes = 0u,
                                          .max_total_bytes = 20u};
  const rund::net::flow::Result zero_limit_rejected =
      rund::net::flow::reserve(rund::net::flow::State{}, zero_limit, 1u);
  TEST_ASSERT(!zero_limit_rejected.ok());
  TEST_ASSERT(zero_limit_rejected.code() ==
              rund::ReasonCode::NetFlowInflightExceeded);
  TEST_ASSERT(std::string_view{zero_limit_rejected.error()} ==
              "net_flow_inflight_exceeded");

  const rund::net::flow::Limit zero_total_limit{.max_inflight_bytes = 20u,
                                                .max_total_bytes = 0u};
  const rund::net::flow::Result zero_total_limit_rejected =
      rund::net::flow::reserve(rund::net::flow::State{}, zero_total_limit, 1u);
  TEST_ASSERT(!zero_total_limit_rejected.ok());
  TEST_ASSERT(zero_total_limit_rejected.code() ==
              rund::ReasonCode::NetFlowTotalExceeded);
  TEST_ASSERT(std::string_view{zero_total_limit_rejected.error()} ==
              "net_flow_total_exceeded");

  const rund::net::flow::State overflow_state{
      .inflight_bytes = 1u,
      .total_bytes = std::numeric_limits<std::uint64_t>::max()};
  const rund::net::flow::Limit overflow_limit{
      .max_inflight_bytes = std::numeric_limits<std::uint64_t>::max(),
      .max_total_bytes = std::numeric_limits<std::uint64_t>::max()};
  const rund::net::flow::Result overflow_rejected =
      rund::net::flow::reserve(overflow_state, overflow_limit, 1u);
  TEST_ASSERT(!overflow_rejected.ok());
  TEST_ASSERT(overflow_rejected.code() ==
              rund::ReasonCode::NetFlowTotalExceeded);
  TEST_ASSERT(std::string_view{overflow_rejected.error()} ==
              "net_flow_total_exceeded");

  const rund::net::flow::State rejected_overflow_state{
      .inflight_bytes = 7u,
      .total_bytes = 7u,
      .rejected_bytes = std::numeric_limits<std::uint64_t>::max()};
  const rund::net::flow::Result rejected_overflow =
      rund::net::flow::reserve(rejected_overflow_state, limit, 4u);
  TEST_ASSERT(!rejected_overflow.ok());
  TEST_ASSERT(rejected_overflow.state.inflight_bytes == 7u);
  TEST_ASSERT(rejected_overflow.state.total_bytes == 7u);
  TEST_ASSERT(rejected_overflow.state.rejected_bytes ==
              std::numeric_limits<std::uint64_t>::max());
  return 0;
}
