#include "local.hpp"
#include <rund/net/listener.hpp>
#include <rund/net/socket.hpp>

#include "test/assert.hpp"

int ListenerInvalidInputsFailClosed() {
  rund::net::Socket invalid{};
  const auto invalid_bind =
      rund::net::bind(invalid.view(), ListenerLoopbackAnyPort());
  TEST_ASSERT(!invalid_bind.ok());
  TEST_ASSERT(invalid_bind.code() == rund::ReasonCode::IoFdInvalid);

  const auto invalid_listen = rund::net::listen(invalid.view(), 64);
  TEST_ASSERT(!invalid_listen.ok());
  TEST_ASSERT(invalid_listen.code() == rund::ReasonCode::IoFdInvalid);

  const auto invalid_local = rund::net::local(invalid.view());
  TEST_ASSERT(!invalid_local.ok());
  TEST_ASSERT(invalid_local.code() == rund::ReasonCode::IoFdInvalid);

  const auto invalid_shutdown =
      rund::net::shutdown(invalid.view(), rund::net::ShutdownMode::ReadWrite);
  TEST_ASSERT(!invalid_shutdown.ok());
  TEST_ASSERT(invalid_shutdown.code() == rund::ReasonCode::IoFdInvalid);

  return 0;
}
