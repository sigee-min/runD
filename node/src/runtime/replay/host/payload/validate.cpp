#include <node/runtime/replay/host/payload.hpp>

namespace rund::node::replay_detail {

bool IsPayloadKind(const ::rund::host::EventKind kind) noexcept {
  return kind == ::rund::host::EventKind::IoRead ||
         kind == ::rund::host::EventKind::IoWrite;
}

bool EventRequiresPayload(const ::rund::host::Event &event) noexcept {
  return IsPayloadKind(event.kind) && event.status == ::rund::host::Status::Ok;
}

bool EventsRequirePayload(
    const std::vector<::rund::host::Event> &events) noexcept {
  for (const ::rund::host::Event &event : events) {
    if (EventRequiresPayload(event)) {
      return true;
    }
  }
  return false;
}

} // namespace rund::node::replay_detail
