#pragma once

#include <rund/hash.hpp>
#include <rund/net/address.hpp>
#include <rund/net/socket.hpp>
#include <rund/net/ready/ticket.hpp>

namespace rund::net::accept {

struct Result : net::Status {
  using net::Status::Status;

  Socket socket{};
  int native_error = 0;
  ::rund::StableHash peer_hash{};
};

[[nodiscard]] Result one(ready::Ticket &&ticket) noexcept;

} // namespace rund::net::accept

namespace rund::net::connect {

struct Result : net::Status {
  using net::Status::Status;

  int native_error = 0;
  ::rund::StableHash address_hash{};
};

[[nodiscard]] Result start(SocketView socket, Address address) noexcept;
[[nodiscard]] Result finish(ready::Ticket &&ticket, Address address) noexcept;

} // namespace rund::net::connect
