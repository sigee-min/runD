#pragma once

#include "../registry/state.hpp"
#include "../socket/access.hpp"

namespace rund::node::test::net {

[[nodiscard]] inline ::rund::net::SocketView
view(const int native, const std::uint64_t generation) noexcept {
  std::shared_lock lock{::rund::net::RegistryGate()};
  ::rund::net::SocketRegistry &sockets = ::rund::net::Registry();
  ::rund::net::SocketSlot *const found = sockets.find(native);
  return found == nullptr ? ::rund::net::SocketView{}
                          : ::rund::net::detail::SocketAccess::view(found,
                                                                    generation);
}

} // namespace rund::node::test::net
