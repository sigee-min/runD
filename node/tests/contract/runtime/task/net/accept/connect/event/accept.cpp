#include "local.hpp"

#include <cstddef>

bool AcceptConnectAcceptEvidenceMatches(const AcceptConnectEventCase &event) {
  std::size_t net_accept_events = 0u;
  bool saw_would_block = false;
  bool saw_success = false;
  for (const rund::host::Event &host_event : event.report.events()) {
    if (host_event.kind != rund::host::EventKind::NetAccept) {
      continue;
    }
    ++net_accept_events;
    saw_would_block =
        saw_would_block || host_event.status == rund::host::Status::WouldBlock;
    saw_success = saw_success || (host_event.status == rund::host::Status::Ok &&
                                  host_event.offset == 3u &&
                                  host_event.payload_hash.value ==
                                      event.accepted.peer_hash.value);
  }
  return net_accept_events >= 2u && saw_would_block && saw_success;
}
