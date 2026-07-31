#include "local.hpp"

bool BasicEventTransferEvidenceMatches(const BasicEventCase &event) {
  const auto &events = event.report.events();
  return events[0].kind == rund::host::EventKind::NetSend &&
         events[1].kind == rund::host::EventKind::NetRecv &&
         events[0].host_handle_id != 0u && events[1].host_handle_id != 0u &&
         events[0].host_handle_id != events[1].host_handle_id &&
         events[0].requested_bytes == event.out.size() &&
         events[0].completed_bytes == event.out.size() &&
         events[1].requested_bytes == event.in.size() &&
         events[1].completed_bytes == event.in.size() &&
         events[0].native_errno == 0 && events[1].native_errno == 0 &&
         events[0].payload_hash.value ==
             rund::host::hash_bytes(event.out.data(), event.out.size()).value &&
         events[1].payload_hash.value ==
             rund::host::hash_bytes(event.in.data(), event.in.size()).value;
}
