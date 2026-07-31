#include <rund/host/event.hpp>

#include "hash/bytes.hpp"
#include "hash/fields.hpp"
#include "replay/layout.hpp"

namespace rund::host {

::rund::StableHash hash_bytes(const std::byte *const data,
                              const std::size_t size) noexcept {
  return node::host_detail::StableByteHash(data, size);
}

::rund::StableHash hash_string(const char *const data,
                               const std::size_t size) noexcept {
  return hash_bytes(reinterpret_cast<const std::byte *>(data), size);
}

::rund::StableHash hash_event(const Event &event) noexcept {
  node::host_detail::StableHashState hash{};
  hash.Mix(event.sequence);
  hash.Mix(static_cast<std::uint16_t>(event.kind));
  hash.Mix(static_cast<std::uint16_t>(event.status));
  hash.Mix(event.task_id);
  hash.Mix(event.logical_time_ns);
  hash.Mix(event.stream_id);
  hash.Mix(event.draw_id);
  hash.Mix(event.host_handle_id);
  hash.Mix(event.offset);
  hash.Mix(event.requested_bytes);
  hash.Mix(event.completed_bytes);
  hash.Mix(static_cast<std::uint32_t>(event.native_errno));
  hash.Mix(event.name_hash.value);
  hash.Mix(event.path_hash.value);
  hash.Mix(event.payload_hash.value);
  return hash.Finish();
}

::rund::StableHash
hash_events(const std::span<const Event> events) noexcept {
  node::host_detail::StableHashState hash{};
  hash.Mix(static_cast<std::uint64_t>(events.size()));
  for (const Event &event : events) {
    hash.Mix(hash_event(event).value);
  }
  return hash.Finish();
}

const char *event_kind_name(const EventKind kind) noexcept {
  return node::host_detail::event_name(kind);
}

} // namespace rund::host
