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
  return std::move(admission).take_socket();
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

// Test-only observation for proving that arbitrary user callbacks execute
// outside every registry generation lease. A nonzero result lets a regression
// fail without calling close() and parking forever on its own reader.
[[nodiscard]] inline std::uint32_t
reader_count(const ::rund::net::SocketView socket) noexcept {
  const ::rund::net::SocketSlot *const slot =
      ::rund::net::detail::SocketAccess::slot(socket);
  return slot == nullptr ? 0u
                         : slot->hot.readers.load(std::memory_order_acquire);
}

} // namespace rund::node::test::net
