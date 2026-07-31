#include "local.hpp"
#include <rund/net/socket.hpp>

#include "test/assert.hpp"

int RunNetLifecycleInvalidCloseCase() {
  rund::net::Socket socket{};
  const rund::net::CloseResult result = socket.close();
  TEST_ASSERT(!result.ok());
  TEST_ASSERT(result.code() == rund::ReasonCode::IoFdInvalid);
  return 0;
}
