#pragma once

#include <rund/hash.hpp>
#include <rund/net/address.hpp>
#include <rund/net/socket.hpp>

namespace rund::net {

struct OpenOptions {
  Family family = Family::IPv4;
  Transport transport = Transport::Stream;
  bool nonblocking = true;
};

struct OpenResult : Status {
  using Status::Status;

  Socket socket{};
  int native_error = 0;
  NonblockingResult nonblocking{};
};

struct BindResult : Status {
  using Status::Status;

  int native_error = 0;
  ::rund::StableHash address_hash{};
};

struct ListenResult : Status {
  using Status::Status;

  int native_error = 0;
  int backlog = 0;
};

struct LocalResult : Status {
  using Status::Status;

  Address address{};
  int native_error = 0;
  ::rund::StableHash address_hash{};
};

enum class ShutdownMode : std::uint8_t {
  Read,
  Write,
  ReadWrite,
};

struct ShutdownResult : Status {
  using Status::Status;

  int native_error = 0;
  ShutdownMode mode = ShutdownMode::ReadWrite;
};

[[nodiscard]] OpenResult open(OpenOptions options = {}) noexcept;
[[nodiscard]] BindResult bind(SocketView socket, Address address) noexcept;
[[nodiscard]] ListenResult listen(SocketView socket, int backlog) noexcept;
[[nodiscard]] LocalResult local(SocketView socket) noexcept;
[[nodiscard]] ShutdownResult shutdown(SocketView socket,
                                      ShutdownMode mode) noexcept;

} // namespace rund::net
