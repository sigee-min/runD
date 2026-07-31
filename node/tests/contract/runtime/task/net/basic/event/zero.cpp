#include "local.hpp"

bool BasicEventZeroEvidenceMatches(const BasicEventCase &event) {
  const auto &events = event.report.events();
  return events[2].kind == rund::host::EventKind::NetSend &&
         events[3].kind == rund::host::EventKind::NetRecv &&
         events[2].host_handle_id == events[0].host_handle_id &&
         events[3].host_handle_id == events[1].host_handle_id &&
         events[2].requested_bytes == 0u && events[2].completed_bytes == 0u &&
         events[3].requested_bytes == 0u && events[3].completed_bytes == 0u &&
         events[2].native_errno == 0 && events[3].native_errno == 0 &&
         events[2].payload_hash.value ==
             rund::host::hash_bytes(event.zero_out.data(), 0u).value &&
         events[3].payload_hash.value ==
             rund::host::hash_bytes(event.zero_in.data(), 0u).value;
}
