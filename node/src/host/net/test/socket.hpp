#pragma once

#include "../registry/state.hpp"
#include "../socket/access.hpp"

#include <utility>

namespace rund::node::test::net {

// Test admission follows the product ownership transfer: once admission
// succeeds, no raw cleanup path retains authority over the descriptor.
[[nodiscard]] inline ::rund::net::Socket admit(int &native) noexcept {
  ::rund::net::SocketAdmission admission =
      ::rund::net::AdmitNativeSocket(native);
  if (admission) {
    native = -1;
  }
  return std::move(admission.socket);
}

[[nodiscard]] inline ::rund::net::Socket admit(int &&native) noexcept {
  return admit(native);
}

[[nodiscard]] inline std::uint64_t generation(const int native) noexcept {
  if (native < 0) {
    return 0u;
  }
  std::shared_lock lock{::rund::net::RegistryGate()};
  const ::rund::net::SocketSlot *const slot =
      ::rund::net::Registry().find(native);
  if (slot == nullptr) {
    return 0u;
  }
  const std::uint64_t current = ::rund::net::registry::load(*slot);
  return ::rund::net::registry::active(current) ? current : 0u;
}

[[nodiscard]] inline int native(const ::rund::net::Socket &socket) noexcept {
  return ::rund::net::detail::SocketAccess::native(socket.view());
}

[[nodiscard]] inline std::uint64_t
generation(const ::rund::net::Socket &socket) noexcept {
  return ::rund::net::detail::SocketAccess::generation(socket.view());
}

} // namespace rund::node::test::net
