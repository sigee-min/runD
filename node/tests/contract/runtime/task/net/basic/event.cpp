#include "event/local.hpp"

#include "test/assert.hpp"

int RunNetBasicEventCase() {
  const BasicEventCase event = RunBasicEventScenario();
  TEST_ASSERT(event.socketpair_ok);
  TEST_ASSERT(event.report.ok());
  TEST_ASSERT(event.send.ok());
  TEST_ASSERT(event.recv.ok());
  TEST_ASSERT(event.zero_send.ok());
  TEST_ASSERT(event.zero_send.bytes == 0);
  TEST_ASSERT(event.zero_recv.ok());
  TEST_ASSERT(event.zero_recv.bytes == 0);
  TEST_ASSERT(event.in == event.out);
  TEST_ASSERT(event.report.events().size() == 4u);
  TEST_ASSERT(BasicEventTransferEvidenceMatches(event));
  TEST_ASSERT(BasicEventZeroEvidenceMatches(event));
  const rund::task::NetworkStats network = event.report.tasks().network();
  TEST_ASSERT(network.recv_calls() == 2u);
  TEST_ASSERT(network.send_calls() == 2u);
  TEST_ASSERT(network.bytes_received() == event.in.size());
  TEST_ASSERT(network.bytes_sent() == event.out.size());
  return 0;
}
