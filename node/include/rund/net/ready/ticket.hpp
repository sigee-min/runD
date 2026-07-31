#pragma once

#include <rund/net/ready/interest.hpp>
#include <rund/net/socket.hpp>

#include <cstdint>
#include <utility>

namespace rund::net::ready {

namespace detail {
struct Access;
}

class Ticket final : public net::Status {
public:
  constexpr Ticket() noexcept = default;
  Ticket(const Ticket &) = delete;
  Ticket &operator=(const Ticket &) = delete;

  constexpr Ticket(Ticket &&other) noexcept
      : net::Status(other.code()),
        socket_(std::exchange(other.socket_, SocketView{})),
        interest_(std::exchange(other.interest_, Interest::Readable)),
        revents_(std::exchange(other.revents_, 0)),
        consumed_(std::exchange(other.consumed_, true)) {}

  constexpr Ticket &operator=(Ticket &&other) noexcept {
    if (this != &other) {
      net::result::Access::set(*this, other.code());
      socket_ = std::exchange(other.socket_, SocketView{});
      interest_ = std::exchange(other.interest_, Interest::Readable);
      revents_ = std::exchange(other.revents_, 0);
      consumed_ = std::exchange(other.consumed_, true);
    }
    return *this;
  }

  [[nodiscard]] constexpr bool ok() const noexcept {
    return ::rund::detail::timed_status_ok(code());
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
  [[nodiscard]] std::string_view error() const noexcept {
    return ::rund::detail::timed_status_error(code());
  }
  [[nodiscard]] constexpr int exit_code() const noexcept {
    return ::rund::detail::timed_status_exit(code());
  }
  [[nodiscard]] constexpr bool ready() const noexcept {
    return code() == ::rund::ReasonCode::Ok && !consumed_;
  }
  [[nodiscard]] constexpr bool timed_out() const noexcept {
    return code() == ::rund::ReasonCode::IoTimedOut;
  }
  [[nodiscard]] constexpr bool consumed() const noexcept { return consumed_; }
  [[nodiscard]] std::uint64_t id() const noexcept {
    return socket_ ? socket_.id() : 0u;
  }
  [[nodiscard]] constexpr Interest interest() const noexcept {
    return interest_;
  }
  [[nodiscard]] constexpr short revents() const noexcept { return revents_; }

private:
  friend struct detail::Access;

  constexpr Ticket(const ::rund::ReasonCode code, const SocketView socket,
                   const Interest interest, const short revents) noexcept
      : net::Status(code), socket_(socket), interest_(interest),
        revents_(revents) {}

  SocketView socket_{};
  Interest interest_ = Interest::Readable;
  short revents_ = 0;
  bool consumed_ = false;
};

} // namespace rund::net::ready
