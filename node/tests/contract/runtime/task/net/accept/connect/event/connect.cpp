#include "local.hpp"

#include <cstddef>

bool AcceptConnectConnectEvidenceMatches(const AcceptConnectEventCase &event) {
  std::size_t net_connect_events = 0u;
  bool saw_address_hash = false;
  for (const rund::host::Event &host_event : event.report.events()) {
    if (host_event.kind != rund::host::EventKind::NetConnect) {
      continue;
    }
    ++net_connect_events;
    saw_address_hash = saw_address_hash || host_event.payload_hash.value ==
                                               event.started.address_hash.value;
  }
  return net_connect_events >= 2u && saw_address_hash;
}
