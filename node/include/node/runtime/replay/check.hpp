#pragma once

#include <rund/replay/code.hpp>

#include <cstdint>

namespace rund::node {

struct RuntimeReplayCheck {
  ::rund::replay::Code code = ::rund::replay::Code::NotChecked;
  std::uint64_t expected_hash = 0u;
  std::uint64_t actual_hash = 0u;

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == ::rund::replay::Code::Ok;
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
  [[nodiscard]] std::string_view error() const noexcept {
    return ::rund::replay::error(code);
  }
};

} // namespace rund::node
