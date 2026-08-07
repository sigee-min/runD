#pragma once

#include <rund/task/cancel.hpp>

namespace rund::detail::task {

class StopAccess final {
public:
  StopAccess() = delete;

  [[nodiscard]] static StopIdentity
  Identity(const ::rund::task::stop_token token) noexcept {
    return token.identity_;
  }
};

} // namespace rund::detail::task
