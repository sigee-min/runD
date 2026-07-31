#pragma once

#include <rund/hash.hpp>
#include <rund/net/address.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>

#include <span>

namespace rund::net::datagram {

struct ReceiveResult : net::Status {
  using net::Status::Status;

  std::int64_t bytes = -1;
  int native_error = 0;
  Address peer{};
  ::rund::StableHash peer_hash{};
};

struct SendResult : net::Status {
  using net::Status::Status;

  std::int64_t bytes = -1;
  int native_error = 0;
  ::rund::StableHash peer_hash{};
};

[[nodiscard]] ReceiveResult receive(ready::Ticket &&ticket,
                                    std::span<std::byte> buffer) noexcept;
[[nodiscard]] SendResult send(ready::Ticket &&ticket,
                              std::span<const std::byte> buffer,
                              Address peer) noexcept;

} // namespace rund::net::datagram
