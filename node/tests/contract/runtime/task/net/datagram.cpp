#include "datagram/local.hpp"

int RunRuntimeTaskNetDatagramContract() {
  TEST_ASSERT(NetDatagramSendReceiveLoopback());
  TEST_ASSERT(NetDatagramWouldBlockAndInvalidInputsFailClosed());
  TEST_ASSERT(NetDatagramReplayEventsAreStable());
  TEST_ASSERT(NetDatagramRejectsOversizedRequest());
  TEST_ASSERT(NetDatagramTransfersEmptyPacket());
  return 0;
}
