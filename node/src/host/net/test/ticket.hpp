#pragma once

#include "../ready/ticket.hpp"

namespace rund::node::test::net {

[[nodiscard]] inline ::rund::net::ready::Ticket
ticket(const ::rund::net::SocketView socket,
       const ::rund::net::ready::Interest interest,
       const short revents = 0) noexcept {
  return ::rund::net::ready::detail::Access::make(::rund::ReasonCode::Ok,
                                                  socket, interest, revents);
}

} // namespace rund::node::test::net
