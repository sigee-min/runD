#include "event/local.hpp"

#include "test/assert.hpp"

int RunAcceptConnectEventCase() {
  const AcceptConnectEventCase event = RunAcceptConnectEventScenario();
  TEST_ASSERT(event.listener_ok);
  TEST_ASSERT(event.listener_nonblocking.ok());
  TEST_ASSERT(event.client_ok);
  TEST_ASSERT(event.client_nonblocking.ok());
  TEST_ASSERT(event.report.ok());
  TEST_ASSERT(!event.empty_accept.ok());
  TEST_ASSERT(event.empty_accept.code() == rund::ReasonCode::IoWouldBlock);
  TEST_ASSERT(event.started.ok());
  TEST_ASSERT(event.accepted.ok());
  TEST_ASSERT(event.finished.ok());
  TEST_ASSERT(AcceptConnectAcceptEvidenceMatches(event));
  TEST_ASSERT(AcceptConnectConnectEvidenceMatches(event));
  return 0;
}
