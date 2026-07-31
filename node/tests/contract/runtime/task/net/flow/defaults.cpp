#include "test/assert.hpp"

#include "local.hpp"

#include <rund/net/flow.hpp>

#include <string_view>

int RunNetFlowDefaultsCase() {
  const rund::net::flow::Limit defaults{};
  TEST_ASSERT(defaults.max_inflight_bytes == 64u * 1024u);
  TEST_ASSERT(defaults.max_total_bytes == 16u * 1024u * 1024u);

  const rund::net::flow::Result not_started{};
  TEST_ASSERT(!not_started.ok());
  TEST_ASSERT(not_started.code() == rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(std::string_view{not_started.error()} == "task_invalid");
  TEST_ASSERT(not_started.state.inflight_bytes == 0u);
  TEST_ASSERT(not_started.state.total_bytes == 0u);
  TEST_ASSERT(not_started.state.rejected_bytes == 0u);
  TEST_ASSERT(not_started.requested_bytes == 0u);
  return 0;
}
