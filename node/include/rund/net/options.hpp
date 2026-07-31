#pragma once

#include <rund/net/socket.hpp>

#include <cstdint>

namespace rund::net::option {

enum class Name : std::uint8_t {
  ReuseAddress,
  ReusePort,
  TcpNoDelay,
  ReceiveBufferBytes,
  SendBufferBytes,
  IPv6Only,
};

struct Value {
  bool flag = false;
  std::int32_t bytes = 0;
};

struct Result : net::Status {
  using net::Status::Status;

  Name option = Name::ReuseAddress;
  Value value{};
  int native_error = 0;
};

[[nodiscard]] Result set(SocketView socket, Name option, Value value) noexcept;
[[nodiscard]] Result get(SocketView socket, Name option) noexcept;

} // namespace rund::net::option
