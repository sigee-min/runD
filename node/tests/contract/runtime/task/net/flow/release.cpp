#include "test/assert.hpp"

#include "local.hpp"

#include <rund/net/flow.hpp>

#include <string_view>

int RunNetFlowReleaseCase() {
  const rund::net::flow::Limit limit{.max_inflight_bytes = 10u,
                                     .max_total_bytes = 20u};
  const rund::net::flow::Result reserved =
      rund::net::flow::reserve(rund::net::flow::State{}, limit, 7u);
  TEST_ASSERT(reserved.ok());

  const rund::net::flow::Result released =
      rund::net::flow::release(reserved.state, 5u);
  TEST_ASSERT(released);
  TEST_ASSERT(released.code() == rund::ReasonCode::Ok);
  TEST_ASSERT(released.error().empty());
  TEST_ASSERT(released.requested_bytes == 5u);
  TEST_ASSERT(released.state.inflight_bytes == 2u);
  TEST_ASSERT(released.state.total_bytes == 7u);

  const rund::net::flow::Result release_exceeded =
      rund::net::flow::release(released.state, 3u);
  TEST_ASSERT(!release_exceeded);
  TEST_ASSERT(release_exceeded.code() ==
              rund::ReasonCode::NetFlowReleaseExceeded);
  TEST_ASSERT(std::string_view{release_exceeded.error()} ==
              "net_flow_release_exceeded");
  TEST_ASSERT(release_exceeded.requested_bytes == 3u);
  TEST_ASSERT(release_exceeded.state.inflight_bytes == 2u);
  TEST_ASSERT(release_exceeded.state.total_bytes == 7u);
  TEST_ASSERT(release_exceeded.state.rejected_bytes ==
              released.state.rejected_bytes + 3u);

  const rund::net::flow::Result zero_release =
      rund::net::flow::release(reserved.state, 0u);
  TEST_ASSERT(!zero_release.ok());
  TEST_ASSERT(zero_release.code() == rund::ReasonCode::NetFlowZeroBytes);
  TEST_ASSERT(std::string_view{zero_release.error()} == "net_flow_zero_bytes");
  TEST_ASSERT(zero_release.state.rejected_bytes ==
              reserved.state.rejected_bytes);
  return 0;
}
