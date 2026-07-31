#pragma once

#include <rund/net/bytes.hpp>
#include <rund/net/datagram.hpp>
#include <rund/net/ready.hpp>

#include <coroutine>
#include <span>
#include <tuple>
#include <utility>

namespace rund::net::datagram::detail {

[[nodiscard]] ready::Wait prepare(SocketView socket,
                                  std::span<std::byte> buffer) noexcept;
[[nodiscard]] ready::Wait prepare(SocketView socket,
                                  std::span<const std::byte> buffer,
                                  Address peer) noexcept;
[[nodiscard]] ReceiveResult consume(ready::Ticket &&ticket,
                                    std::span<std::byte> buffer) noexcept;
[[nodiscard]] SendResult consume(ready::Ticket &&ticket,
                                 std::span<const std::byte> buffer,
                                 Address peer) noexcept;

} // namespace rund::net::datagram::detail

namespace rund::net::detail {

[[nodiscard]] ready::Wait prepare(SocketView socket,
                                  std::span<std::byte> buffer) noexcept;
[[nodiscard]] ready::Wait prepare(SocketView socket,
                                  std::span<const std::byte> buffer) noexcept;
[[nodiscard]] ReceiveResult consume(ready::Ticket &&ticket,
                                    std::span<std::byte> buffer) noexcept;
[[nodiscard]] SendResult consume(ready::Ticket &&ticket,
                                 std::span<const std::byte> buffer) noexcept;

inline constexpr auto ReceiveBytes = static_cast<ReceiveResult (*)(
    ready::Ticket &&, std::span<std::byte>) noexcept>(&detail::consume);
inline constexpr auto SendBytes = static_cast<SendResult (*)(
    ready::Ticket &&, std::span<const std::byte>) noexcept>(&detail::consume);
inline constexpr auto ReceiveDatagram = static_cast<datagram::ReceiveResult (*)(
    ready::Ticket &&, std::span<std::byte>) noexcept>(
    &::rund::net::datagram::detail::consume);
inline constexpr auto SendDatagram = static_cast<datagram::SendResult (*)(
    ready::Ticket &&, std::span<const std::byte>, Address) noexcept>(
    &::rund::net::datagram::detail::consume);
inline constexpr auto PrepareReceiveBytes =
    static_cast<ready::Wait (*)(SocketView, std::span<std::byte>) noexcept>(
        &detail::prepare);
inline constexpr auto PrepareSendBytes = static_cast<ready::Wait (*)(
    SocketView, std::span<const std::byte>) noexcept>(&detail::prepare);

template <class Result, auto Consume, class... Args> class ReadyAwaiter final {
public:
  ReadyAwaiter(ready::Wait readiness, std::tuple<Args...> args) noexcept
      : readiness_(std::move(readiness)), args_(std::move(args)) {}

  [[nodiscard]] bool await_ready() const noexcept {
    return readiness_.await_ready();
  }

  bool await_suspend(const std::coroutine_handle<> handle) noexcept {
    return readiness_.await_suspend(handle);
  }

  [[nodiscard]] Result await_resume() noexcept {
    ready::Ticket ticket = readiness_.await_resume();
    return std::apply(
        [&ticket](auto &&...args) noexcept {
          return Consume(std::move(ticket),
                         std::forward<decltype(args)>(args)...);
        },
        std::move(args_));
  }

private:
  ready::Awaiter readiness_;
  std::tuple<Args...> args_;
};

template <class Result, auto Prepare, auto Consume, class... Args>
class ReadyOperation final {
public:
  ReadyOperation(const SocketView socket, Args... args) noexcept
      : socket_(socket), args_(std::move(args)...) {}

  ReadyOperation(const ReadyOperation &) = delete;
  ReadyOperation &operator=(const ReadyOperation &) = delete;
  ReadyOperation(ReadyOperation &&source) noexcept
      : socket_(std::exchange(source.socket_, SocketView{})),
        args_(std::move(source.args_)) {}
  ReadyOperation &operator=(ReadyOperation &&source) noexcept {
    if (this != &source) {
      socket_ = std::exchange(source.socket_, SocketView{});
      args_ = std::move(source.args_);
    }
    return *this;
  }

  [[nodiscard]] ReadyAwaiter<Result, Consume, Args...>
  operator co_await() && noexcept {
    const SocketView socket = std::exchange(socket_, SocketView{});
    ready::Wait readiness = std::apply(
        [socket](const auto &...args) noexcept {
          return Prepare(socket, args...);
        },
        args_);
    return ReadyAwaiter<Result, Consume, Args...>{std::move(readiness),
                                                  std::move(args_)};
  }

private:
  SocketView socket_;
  std::tuple<Args...> args_;
};

} // namespace rund::net::detail

namespace rund::net {

class Receive final {
public:
  Receive(const Receive &) = delete;
  Receive &operator=(const Receive &) = delete;
  Receive(Receive &&) noexcept = default;
  Receive &operator=(Receive &&) noexcept = default;

  [[nodiscard]] auto operator co_await() && noexcept {
    return std::move(operation_).operator co_await();
  }

private:
  using Buffer = std::span<std::byte>;
  using Operation =
      detail::ReadyOperation<ReceiveResult, detail::PrepareReceiveBytes,
                             detail::ReceiveBytes, Buffer>;

  friend Receive receive(SocketView, std::span<std::byte>) noexcept;

  Receive(const SocketView socket, const Buffer buffer) noexcept
      : operation_(socket, buffer) {}

  Operation operation_;
};

class Send final {
public:
  Send(const Send &) = delete;
  Send &operator=(const Send &) = delete;
  Send(Send &&) noexcept = default;
  Send &operator=(Send &&) noexcept = default;

  [[nodiscard]] auto operator co_await() && noexcept {
    return std::move(operation_).operator co_await();
  }

private:
  using Buffer = std::span<const std::byte>;
  using Operation = detail::ReadyOperation<SendResult, detail::PrepareSendBytes,
                                           detail::SendBytes, Buffer>;

  friend Send send(SocketView, std::span<const std::byte>) noexcept;

  Send(const SocketView socket, const Buffer buffer) noexcept
      : operation_(socket, buffer) {}

  Operation operation_;
};

[[nodiscard]] inline Receive
receive(const SocketView socket, const std::span<std::byte> buffer) noexcept {
  return Receive{socket, buffer};
}

[[nodiscard]] inline Send
send(const SocketView socket,
     const std::span<const std::byte> buffer) noexcept {
  return Send{socket, buffer};
}

} // namespace rund::net

namespace rund::net::datagram {

namespace detail {

inline constexpr auto PrepareReceive =
    static_cast<ready::Wait (*)(SocketView, std::span<std::byte>) noexcept>(
        &detail::prepare);
inline constexpr auto PrepareSend =
    static_cast<ready::Wait (*)(SocketView, std::span<const std::byte>,
                                Address) noexcept>(&detail::prepare);

} // namespace detail

class Receive final {
public:
  Receive(const Receive &) = delete;
  Receive &operator=(const Receive &) = delete;
  Receive(Receive &&) noexcept = default;
  Receive &operator=(Receive &&) noexcept = default;

  [[nodiscard]] auto operator co_await() && noexcept {
    return std::move(operation_).operator co_await();
  }

private:
  using Buffer = std::span<std::byte>;
  using Operation =
      net::detail::ReadyOperation<ReceiveResult, detail::PrepareReceive,
                                  net::detail::ReceiveDatagram, Buffer>;

  friend Receive receive(SocketView, std::span<std::byte>) noexcept;

  Receive(const SocketView socket, const Buffer buffer) noexcept
      : operation_(socket, buffer) {}

  Operation operation_;
};

class Send final {
public:
  Send(const Send &) = delete;
  Send &operator=(const Send &) = delete;
  Send(Send &&) noexcept = default;
  Send &operator=(Send &&) noexcept = default;

  [[nodiscard]] auto operator co_await() && noexcept {
    return std::move(operation_).operator co_await();
  }

private:
  using Buffer = std::span<const std::byte>;
  using Operation =
      net::detail::ReadyOperation<SendResult, detail::PrepareSend,
                                  net::detail::SendDatagram, Buffer, Address>;

  friend Send send(SocketView, std::span<const std::byte>, Address) noexcept;

  Send(const SocketView socket, const Buffer buffer,
       const Address peer) noexcept
      : operation_(socket, buffer, peer) {}

  Operation operation_;
};

[[nodiscard]] inline Receive
receive(const SocketView socket, const std::span<std::byte> buffer) noexcept {
  return Receive{socket, buffer};
}

[[nodiscard]] inline Send send(const SocketView socket,
                               const std::span<const std::byte> buffer,
                               const Address peer) noexcept {
  return Send{socket, buffer, peer};
}

} // namespace rund::net::datagram
