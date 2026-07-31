#include "layout.hpp"

#include <array>
#include <cstddef>
#include <type_traits>

namespace rund::node::host_detail {
namespace {

inline constexpr auto kEventNames = std::to_array<const char *>({
    "random_draw",
    "logical_clock_read",
    "timer_sleep",
    "io_ready",
    "io_read",
    "io_write",
    "io_close",
    "env_get",
    "net_recv",
    "net_send",
    "io_set_nonblocking",
    "net_accept",
    "net_connect",
    "net_socket",
    "net_bind",
    "net_listen",
    "net_local_address",
    "net_shutdown",
    "net_recv_datagram",
    "net_send_datagram",
    "net_set_socket_option",
    "net_get_socket_option",
    "net_recv_vectored",
    "net_send_vectored",
});

using EventKindValue = std::underlying_type_t<::rund::host::EventKind>;

static_assert(static_cast<std::size_t>(::rund::host::EventKind::NetSendVectored) ==
              kEventNames.size());

[[nodiscard]] constexpr std::size_t
event_index(const ::rund::host::EventKind kind) noexcept {
  const auto value = static_cast<EventKindValue>(kind);
  return value == 0u || value > kEventNames.size()
             ? kEventNames.size()
             : static_cast<std::size_t>(value - 1u);
}

} // namespace

bool known_replay_kind(const ::rund::host::EventKind kind) noexcept {
  return event_index(kind) < kEventNames.size();
}

const char *event_name(const ::rund::host::EventKind kind) noexcept {
  if (kind == ::rund::host::EventKind::None) {
    return "none";
  }
  const std::size_t index = event_index(kind);
  return index < kEventNames.size() ? kEventNames[index] : "unknown";
}

} // namespace rund::node::host_detail
