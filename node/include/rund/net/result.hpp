#pragma once

#include <rund/reason.hpp>

#include <string_view>

namespace rund::net {

namespace result {
struct Access;
}

class Status {
public:
  constexpr Status() noexcept = default;
  explicit constexpr Status(const ::rund::ReasonCode code) noexcept
      : code_(code) {}

  [[nodiscard]] constexpr ::rund::ReasonCode code() const noexcept {
    return code_;
  }
  [[nodiscard]] constexpr bool ok() const noexcept {
    return ::rund::detail::status_ok(code_);
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
  [[nodiscard]] std::string_view error() const noexcept {
    return ::rund::detail::status_error(code_);
  }
  [[nodiscard]] constexpr int exit_code() const noexcept {
    return ::rund::detail::status_exit(code_);
  }

private:
  friend struct result::Access;
  ::rund::ReasonCode code_ = ::rund::ReasonCode::TaskInvalid;
};

namespace result {

struct Access final {
  static constexpr void set(Status &status,
                            const ::rund::ReasonCode code) noexcept {
    status.code_ = code;
  }
};

} // namespace result
} // namespace rund::net
