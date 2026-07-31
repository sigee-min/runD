#include "test/assert.hpp"

#include "local.hpp"

#include <rund/net/flow.hpp>

#include <string_view>

int RunNetFlowReserveCase() {
  const rund::net::flow::Limit limit{.max_inflight_bytes = 10u,
                                     .max_total_bytes = 20u};
  const rund::net::flow::Result reserved =
      rund::net::flow::reserve(rund::net::flow::State{}, limit, 7u);
  TEST_ASSERT(reserved);
  TEST_ASSERT(reserved.ok());
  TEST_ASSERT(reserved.code() == rund::ReasonCode::Ok);
  TEST_ASSERT(reserved.error().empty());
  TEST_ASSERT(reserved.requested_bytes == 7u);
  TEST_ASSERT(reserved.state.inflight_bytes == 7u);
  TEST_ASSERT(reserved.state.total_bytes == 7u);
  TEST_ASSERT(reserved.state.rejected_bytes == 0u);

  const rund::net::flow::Result inflight_exceeded =
      rund::net::flow::reserve(reserved.state, limit, 4u);
  TEST_ASSERT(!inflight_exceeded);
  TEST_ASSERT(inflight_exceeded.code() ==
              rund::ReasonCode::NetFlowInflightExceeded);
  TEST_ASSERT(std::string_view{inflight_exceeded.error()} ==
              "net_flow_inflight_exceeded");
  TEST_ASSERT(inflight_exceeded.requested_bytes == 4u);
  TEST_ASSERT(inflight_exceeded.state.inflight_bytes == 7u);
  TEST_ASSERT(inflight_exceeded.state.total_bytes == 7u);
  TEST_ASSERT(inflight_exceeded.state.rejected_bytes == 4u);

  const rund::net::flow::State near_total{
      .inflight_bytes = 2u, .total_bytes = 19u, .rejected_bytes = 5u};
  const rund::net::flow::Result total_exceeded =
      rund::net::flow::reserve(near_total, limit, 2u);
  TEST_ASSERT(!total_exceeded);
  TEST_ASSERT(total_exceeded.code() == rund::ReasonCode::NetFlowTotalExceeded);
  TEST_ASSERT(std::string_view{total_exceeded.error()} ==
              "net_flow_total_exceeded");
  TEST_ASSERT(total_exceeded.requested_bytes == 2u);
  TEST_ASSERT(total_exceeded.state.inflight_bytes == 2u);
  TEST_ASSERT(total_exceeded.state.total_bytes == 19u);
  TEST_ASSERT(total_exceeded.state.rejected_bytes ==
              near_total.rejected_bytes + 2u);

  const rund::net::flow::Result send_record =
      rund::net::flow::record_send(rund::net::flow::State{}, limit, 3u);
  TEST_ASSERT(send_record.ok());
  TEST_ASSERT(send_record.state.inflight_bytes == 3u);
  TEST_ASSERT(send_record.state.total_bytes == 3u);

  const rund::net::flow::Result receive_record =
      rund::net::flow::record_receive(send_record.state, limit, 2u);
  TEST_ASSERT(receive_record.ok());
  TEST_ASSERT(receive_record.state.inflight_bytes == 5u);
  TEST_ASSERT(receive_record.state.total_bytes == 5u);
  return 0;
}
