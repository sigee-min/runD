#pragma once

#include "../../net.hpp"

#include <cerrno>

namespace rund::node {

[[nodiscard]] inline NativeCallResult
PosixNetResult(const std::int64_t value, const int error,
               const bool invalid_input = false) noexcept {
  bool would_block = false;
#if defined(EAGAIN)
  would_block = error == EAGAIN;
#endif
#if defined(EWOULDBLOCK) && (!defined(EAGAIN) || EWOULDBLOCK != EAGAIN)
  would_block = would_block || error == EWOULDBLOCK;
#endif
  bool in_progress = false;
#if defined(EINPROGRESS)
  in_progress = error == EINPROGRESS;
#endif
  const NativeCallState state =
      invalid_input ? NativeCallState::InvalidInput
      : in_progress ? NativeCallState::InProgress
      : would_block ? NativeCallState::WouldBlock
      : value >= 0 ? NativeCallState::Complete
                   : NativeCallState::Failed;
  return NativeCallResult{.value = value, .error = error, .state = state};
}

[[nodiscard]] inline NativeAddressResult
PosixAddressResult(const std::int64_t value, const int error,
                   const ::rund::net::Address& address = {}) noexcept {
  return NativeAddressResult{
      .call = PosixNetResult(value, error),
      .address = value < 0 ? ::rund::net::Address{} : address,
  };
}

} // namespace rund::node
