#pragma once

#include "src/host/net/registry/state.hpp"

#include <rund/net/socket.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::node::test_contract::net_registry_lifetime {

[[nodiscard]] inline bool Seed(const rund::net::SocketView socket,
                               const std::uint64_t generation) noexcept {
  const int native = rund::net::detail::SocketAccess::native(socket);
  std::lock_guard lock{rund::net::RegistryGate()};
  rund::net::SocketRegistry &sockets = rund::net::Registry();
  rund::net::SocketSlot *const found = sockets.find(native);
  if (found == nullptr) {
    return false;
  }
  found->hot.generation.store(generation, std::memory_order_release);
  return true;
}

[[nodiscard]] inline std::uint64_t
Read(const rund::net::SocketView socket) noexcept {
  const rund::net::SocketSlot *const slot =
      rund::net::detail::SocketAccess::slot(socket);
  return slot == nullptr ? 0u : rund::net::registry::load(*slot);
}

[[nodiscard]] inline bool Retire(const rund::net::SocketView socket) noexcept {
  rund::net::SocketSlot *const slot =
      rund::net::detail::SocketAccess::slot(socket);
  if (slot == nullptr) {
    return false;
  }
  std::lock_guard lock{rund::net::RegistryGate()};
  if (rund::net::registry::load(*slot) !=
      rund::net::detail::SocketAccess::generation(socket)) {
    return false;
  }
  rund::net::registry::retire(*slot);
  return true;
}

[[nodiscard]] inline rund::net::SocketRegistryStats Stats() noexcept {
  std::shared_lock lock{rund::net::RegistryGate()};
  return rund::net::Registry().stats();
}

[[nodiscard]] inline bool
HasReservation(const rund::net::SocketView socket) noexcept {
  std::shared_lock lock{rund::net::RegistryGate()};
  const rund::net::SocketSlot *const slot =
      rund::net::detail::SocketAccess::slot(socket);
  return slot != nullptr && slot->active_owner.active();
}

[[nodiscard]] inline std::shared_ptr<std::atomic<std::uint32_t>>
ReservationCounter(const rund::net::SocketView socket) noexcept {
  std::shared_lock lock{rund::net::RegistryGate()};
  const rund::net::SocketSlot *const slot =
      rund::net::detail::SocketAccess::slot(socket);
  return slot == nullptr ? nullptr : slot->active_owner.live_entries;
}

} // namespace rund::node::test_contract::net_registry_lifetime
