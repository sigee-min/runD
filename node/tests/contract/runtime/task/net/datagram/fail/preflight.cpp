#include "preflight/local.hpp"

bool NetDatagramPreflightFailuresFailClosed() {
  DATAGRAM_ASSERT(NetDatagramRejectsWouldBlockPreflight());
  DATAGRAM_ASSERT(NetDatagramRejectsNullPreflight());
  return true;
}
