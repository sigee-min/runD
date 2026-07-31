#pragma once

#include <rund/net/socket.hpp>

namespace rund::net {

struct SocketSlot;

} // namespace rund::net

namespace rund::net::detail {

struct SocketAccess final {
  [[nodiscard]] static Socket make(SocketSlot *const slot,
                                   const std::uint64_t generation) noexcept {
    return Socket{slot, generation};
  }

  [[nodiscard]] static constexpr SocketView
  view(SocketSlot *const slot, const std::uint64_t generation) noexcept {
    return SocketView{slot, generation};
  }

  [[nodiscard]] static constexpr SocketSlot *
  slot(const SocketView socket) noexcept {
    return static_cast<SocketSlot *>(socket.slot_);
  }
  [[nodiscard]] static constexpr std::uint64_t
  generation(const SocketView socket) noexcept {
    return socket.generation_;
  }
  [[nodiscard]] static constexpr std::uint64_t id(const int native) noexcept {
    return native < 0 ? 0u : static_cast<std::uint64_t>(native) + 1u;
  }
  [[nodiscard]] static int native(SocketView socket) noexcept;
};

} // namespace rund::net::detail
