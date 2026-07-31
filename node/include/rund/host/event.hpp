#pragma once

#include <rund/host/hash.hpp>
#include <rund/host/status.hpp>

#include <cstdint>
#include <span>

namespace rund::host {

enum class EventKind : std::uint16_t {
  None = 0u,
  RandomDraw = 1u,
  LogicalClockRead = 2u,
  TimerSleep = 3u,
  IoReady = 4u,
  IoRead = 5u,
  IoWrite = 6u,
  IoClose = 7u,
  EnvGet = 8u,
  NetRecv = 9u,
  NetSend = 10u,
  IoSetNonblocking = 11u,
  NetAccept = 12u,
  NetConnect = 13u,
  NetSocket = 14u,
  NetBind = 15u,
  NetListen = 16u,
  NetLocalAddress = 17u,
  NetShutdown = 18u,
  NetRecvDatagram = 19u,
  NetSendDatagram = 20u,
  NetSetSocketOption = 21u,
  NetGetSocketOption = 22u,
  NetRecvVectored = 23u,
  NetSendVectored = 24u,
};

struct Event final {
  std::uint64_t sequence = 0u;
  EventKind kind = EventKind::None;
  Status status = Status::Invalid;
  std::uint64_t task_id = 0u;
  std::uint64_t logical_time_ns = 0u;
  std::uint64_t stream_id = 0u;
  std::uint64_t draw_id = 0u;
  std::uint64_t host_handle_id = 0u;
  std::uint64_t offset = 0u;
  std::uint64_t requested_bytes = 0u;
  std::uint64_t completed_bytes = 0u;
  std::int32_t native_errno = 0;
  StableHash name_hash{};
  StableHash path_hash{};
  StableHash payload_hash{};
};

[[nodiscard]] StableHash hash_event(const Event &event) noexcept;
[[nodiscard]] StableHash hash_events(std::span<const Event> events) noexcept;
[[nodiscard]] const char *event_kind_name(EventKind kind) noexcept;

} // namespace rund::host
