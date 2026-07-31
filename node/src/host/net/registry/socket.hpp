#pragma once

#include <rund/net/socket.hpp>
#include <rund/reason.hpp>

#include "../socket/access.hpp"

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

struct SocketAdmission {
  Socket socket{};
  ::rund::ReasonCode code = ::rund::ReasonCode::TaskInvalid;

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return code == ::rund::ReasonCode::Ok;
  }
};

[[nodiscard]] bool IsCurrentSocket(SocketView socket) noexcept;
[[nodiscard]] SocketLease LeaseSocket(SocketView socket) noexcept;
[[nodiscard]] SocketAdmission AdmitNativeSocket(int native_socket) noexcept;
[[nodiscard]] bool BeginSocketClose(SocketView socket) noexcept;
void FinishSocketClose(SocketView socket) noexcept;

} // namespace rund::net
