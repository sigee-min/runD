#pragma once

#include <rund/net/ready/ticket.hpp>

#include <cstdint>
#include <span>

namespace rund::net {

struct ReceiveResult : Status {
  using Status::Status;

  std::int64_t bytes = -1;
  int native_error = 0;
};

struct SendResult : Status {
  using Status::Status;

  std::int64_t bytes = -1;
  int native_error = 0;
};

namespace direct {

[[nodiscard]] ReceiveResult receive(SocketView socket,
                                    std::span<std::byte> buffer) noexcept;
[[nodiscard]] SendResult send(SocketView socket,
                              std::span<const std::byte> buffer) noexcept;

} // namespace direct

[[nodiscard]] ReceiveResult receive(ready::Ticket &&ticket,
                                    std::span<std::byte> buffer) noexcept;
[[nodiscard]] SendResult send(ready::Ticket &&ticket,
                              std::span<const std::byte> buffer) noexcept;

} // namespace rund::net
