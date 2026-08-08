#pragma once

#include <rund/net/socket.hpp>
#include <rund/reason.hpp>

#include "../socket/access.hpp"

#include <cstdlib>
#include <utility>

namespace rund::net {

class SocketLease final {
public:
  SocketLease() noexcept = default;
  ~SocketLease() noexcept;

  SocketLease(const SocketLease &) = delete;
  SocketLease &operator=(const SocketLease &) = delete;
  SocketLease(SocketLease &&other) noexcept;
  SocketLease &operator=(SocketLease &&other) noexcept;

  [[nodiscard]] explicit operator bool() const noexcept {
    return slot_ != nullptr;
  }
  [[nodiscard]] int native() const noexcept { return native_; }
  [[nodiscard]] std::uint64_t id() const noexcept {
    return detail::SocketAccess::id(native_);
  }

private:
  friend SocketLease LeaseSocket(SocketView) noexcept;

  SocketLease(SocketSlot *slot, int native) noexcept
      : slot_(slot), native_(native) {}
  void release() noexcept;

  SocketSlot *slot_ = nullptr;
  int native_ = -1;
};

class SocketAdmission final {
public:
  SocketAdmission() = delete;
  SocketAdmission(const SocketAdmission &) = delete;
  SocketAdmission &operator=(const SocketAdmission &) = delete;

  SocketAdmission(SocketAdmission &&other) noexcept
      : socket_(std::move(other.socket_)),
        code_(std::exchange(other.code_, ::rund::ReasonCode::TaskInvalid)) {}

  SocketAdmission &operator=(SocketAdmission &&other) noexcept {
    if (this != &other) {
      socket_ = std::move(other.socket_);
      code_ = std::exchange(other.code_, ::rund::ReasonCode::TaskInvalid);
    }
    return *this;
  }

  [[nodiscard]] static SocketAdmission
  failure(const ::rund::ReasonCode code) noexcept {
    if (code == ::rund::ReasonCode::Ok) {
      std::abort();
    }
    return SocketAdmission{Socket{}, code};
  }

  [[nodiscard]] static SocketAdmission success(Socket socket) noexcept {
    if (!socket) {
      std::abort();
    }
    return SocketAdmission{std::move(socket), ::rund::ReasonCode::Ok};
  }

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return code_ == ::rund::ReasonCode::Ok;
  }

  [[nodiscard]] constexpr ::rund::ReasonCode code() const noexcept {
    return code_;
  }

  [[nodiscard]] Socket take_socket() && noexcept {
    const ::rund::ReasonCode code =
        std::exchange(code_, ::rund::ReasonCode::TaskInvalid);
    if (code != ::rund::ReasonCode::Ok) {
      return Socket{};
    }
    return std::move(socket_);
  }

private:
  SocketAdmission(Socket socket, const ::rund::ReasonCode code) noexcept
      : socket_(std::move(socket)), code_(code) {}

  Socket socket_{};
  ::rund::ReasonCode code_ = ::rund::ReasonCode::TaskInvalid;
};

[[nodiscard]] bool IsCurrentSocket(SocketView socket) noexcept;
[[nodiscard]] SocketLease LeaseSocket(SocketView socket) noexcept;
[[nodiscard]] SocketAdmission AdmitNativeSocket(int native_socket) noexcept;
[[nodiscard]] bool BeginSocketClose(SocketView socket) noexcept;
void FinishSocketClose(SocketView socket) noexcept;

} // namespace rund::net
