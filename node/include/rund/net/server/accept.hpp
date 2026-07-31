#pragma once

#include <rund/net/accept.hpp>
#include <rund/net/handoff.hpp>
#include <rund/net/server/options.hpp>
#include <rund/net/server/peer.hpp>
#include <rund/net/server/result.hpp>

#include <utility>

namespace rund::net::server::detail {

struct Prepared : net::Status {
  using net::Status::Status;

  Peer peer{};
};

[[nodiscard]] net::accept::Result next(net::SocketView listener) noexcept;

[[nodiscard]] inline Prepared
prepare(net::accept::Result &&accepted,
        const net::accept::Options options) noexcept {
  net::accept::Prepared prepared =
      net::accept::prepare(std::move(accepted), options);
  if (!prepared) {
    return Prepared{prepared.code()};
  }
  Prepared result{::rund::ReasonCode::Ok};
  result.peer.socket = std::move(prepared.socket);
  return result;
}

} // namespace rund::net::server::detail
