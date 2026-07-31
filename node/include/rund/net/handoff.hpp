#pragma once

#include <rund/net/connection.hpp>
#include <rund/net/socket.hpp>

namespace rund::net::accept {

struct Options {
  bool nonblocking = true;
};

struct Prepared : net::Status {
  using net::Status::Status;

  Socket socket{};
  NonblockingResult nonblocking{};
};

[[nodiscard]] Prepared prepare(Result &&accepted,
                               Options options = {}) noexcept;

} // namespace rund::net::accept
