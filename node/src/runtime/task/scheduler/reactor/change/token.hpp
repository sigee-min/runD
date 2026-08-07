#pragma once

#include "../../../../reactor/readiness/state.hpp"

#include <cstdint>

namespace rund::node {

class ReactorInvalidChangeToken final {
public:
  [[nodiscard]] static constexpr ReactorInvalidChangeToken none() noexcept {
    return ReactorInvalidChangeToken{kInvalidReactorHandle, 0u};
  }

  [[nodiscard]] static constexpr ReactorInvalidChangeToken
  observed(const ReactorHandle handle,
           const std::uint64_t fd_generation) noexcept {
    return handle == kInvalidReactorHandle
               ? none()
               : ReactorInvalidChangeToken{handle, fd_generation};
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return handle_ != kInvalidReactorHandle;
  }

  [[nodiscard]] constexpr ReactorHandle handle() const noexcept {
    return handle_;
  }

  [[nodiscard]] constexpr std::uint64_t fd_generation() const noexcept {
    return fd_generation_;
  }

private:
  constexpr ReactorInvalidChangeToken(
      const ReactorHandle handle,
      const std::uint64_t fd_generation) noexcept
      : handle_(handle), fd_generation_(fd_generation) {}

  ReactorHandle handle_;
  std::uint64_t fd_generation_;
};

} // namespace rund::node
