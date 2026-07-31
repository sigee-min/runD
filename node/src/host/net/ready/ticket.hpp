#pragma once

#include <rund/net/ready.hpp>

#include "../registry/socket.hpp"

namespace rund::net::ready::detail {

struct Claim;

[[nodiscard]] Wait fail(SocketView socket, Interest interest,
                        ::rund::ReasonCode code) noexcept;
[[nodiscard]] Wait complete(SocketView socket, Interest interest) noexcept;

struct Access final {
  [[nodiscard]] static constexpr Ticket make(const ::rund::ReasonCode code,
                                             const SocketView socket,
                                             const Interest interest,
                                             const short revents = 0) noexcept {
    return Ticket{code, socket, interest, revents};
  }

  [[nodiscard]] static constexpr SocketView
  socket(const Ticket &ticket) noexcept {
    return ticket.socket_;
  }

  [[nodiscard]] static constexpr bool consume(Ticket &ticket) noexcept {
    return std::exchange(ticket.consumed_, true);
  }

  [[nodiscard]] static Wait wait(const task::IoOp operation,
                                 const SocketView socket,
                                 const Interest interest,
                                 const short native_interest,
                                 const bool deferred) noexcept {
    Wait wait{};
    wait.operation_ = operation;
    wait.socket_ = socket;
    wait.interest_ = interest;
    wait.native_interest_ = native_interest;
    wait.deferred_ = deferred;
    return wait;
  }
};

class Operation final {
public:
  [[nodiscard]] explicit operator bool() const noexcept {
    return code_ == ::rund::ReasonCode::Ok && static_cast<bool>(lease_);
  }
  [[nodiscard]] ::rund::ReasonCode code() const noexcept { return code_; }
  [[nodiscard]] int native() const noexcept { return lease_.native(); }
  [[nodiscard]] std::uint64_t id() const noexcept { return lease_.id(); }

private:
  friend Operation prepare(const Claim &) noexcept;

  SocketLease lease_{};
  ::rund::ReasonCode code_ = ::rund::ReasonCode::TaskInvalid;
};

// Claim is the allocation-free ownership transition for a public Ticket&&.
// It invalidates the caller's capability before operation-specific preflight,
// while deferring the socket generation lease until native work is possible.
struct Claim final {
  SocketView socket{};
  ::rund::ReasonCode code = ::rund::ReasonCode::TaskInvalid;

  [[nodiscard]] explicit operator bool() const noexcept {
    return code == ::rund::ReasonCode::Ok;
  }
};

[[nodiscard]] inline bool Supports(const Interest actual,
                                   const Interest required) noexcept {
  return actual == Interest::ReadWrite || actual == required;
}

[[nodiscard]] inline Claim claim(Ticket &&ticket,
                                 const Interest required) noexcept {
  if (Access::consume(ticket)) {
    return Claim{.code = ::rund::ReasonCode::NetTicketConsumed};
  }
  if (ticket.code() != ::rund::ReasonCode::Ok) {
    return Claim{.code = ticket.code()};
  }
  if (!Supports(ticket.interest(), required)) {
    return Claim{.code = ::rund::ReasonCode::NetTicketInterestMismatch};
  }
  return Claim{.socket = Access::socket(ticket),
               .code = ::rund::ReasonCode::Ok};
}

[[nodiscard]] inline Operation prepare(const Claim &claim) noexcept {
  SocketLease lease = LeaseSocket(claim.socket);
  if (!lease) {
    Operation operation{};
    operation.code_ = ::rund::ReasonCode::IoFdInvalid;
    return operation;
  }
  Operation operation{};
  operation.lease_ = std::move(lease);
  operation.code_ = ::rund::ReasonCode::Ok;
  return operation;
}

} // namespace rund::net::ready::detail
