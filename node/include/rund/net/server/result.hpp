#pragma once

#include <rund/net/result.hpp>

#include <cstdint>
#include <string_view>

namespace rund::net::server {

class PeerResult final : public net::Status {
public:
  constexpr PeerResult() noexcept
      : net::Status(::rund::ReasonCode::NetPeerHandlerFailed) {}

  [[nodiscard]] static constexpr PeerResult complete() noexcept {
    return PeerResult{::rund::ReasonCode::Ok};
  }
  [[nodiscard]] static constexpr PeerResult stop() noexcept {
    return PeerResult{::rund::ReasonCode::NetPeerHandlerStopped};
  }
  [[nodiscard]] static PeerResult fail(const ::rund::ReasonCode code) noexcept {
    return PeerResult{::rund::ValidReasonCode(code) &&
                              code != ::rund::ReasonCode::Ok &&
                              code != ::rund::ReasonCode::NetPeerHandlerStopped
                          ? code
                          : ::rund::ReasonCode::NetPeerHandlerFailed};
  }
  [[nodiscard]] constexpr bool stopped() const noexcept {
    return code() == ::rund::ReasonCode::NetPeerHandlerStopped;
  }

private:
  explicit constexpr PeerResult(const ::rund::ReasonCode code) noexcept
      : net::Status(code) {}
};

struct Result : net::Status {
  using Status::Status;

  bool would_block = false;
  bool budget_exhausted = false;
  std::uint32_t accepted = 0u;
  std::uint32_t started = 0u;
  std::uint32_t completed = 0u;
  std::uint32_t failed = 0u;
  std::uint32_t stopped = 0u;
  std::uint32_t rejected = 0u;
};

} // namespace rund::net::server
