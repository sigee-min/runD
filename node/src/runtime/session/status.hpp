#pragma once

#include <rund/session.hpp>

namespace rund::detail::session {

struct StatusAccess final {
  [[nodiscard]] static constexpr ::rund::Session::Status
  make(const ReasonCode code, const ::rund::SessionState state) noexcept {
    return ::rund::Session::Status{code, state};
  }
};

} // namespace rund::detail::session
