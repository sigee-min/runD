#pragma once

#include <cstdint>
#include <string_view>

namespace rund {

enum class ReasonCode : std::uint16_t {
#define RUND_NODE_REASON(value, name, text, category) name = value,
#include <rund/reason.def>
#undef RUND_NODE_REASON
};

[[nodiscard]] const char *ReasonString(ReasonCode code) noexcept;
[[nodiscard]] bool ValidReasonCode(ReasonCode code) noexcept;
[[nodiscard]] bool ValidPreparedMemoryReason(ReasonCode code) noexcept;

namespace detail {

[[nodiscard]] constexpr bool status_ok(const ReasonCode code) noexcept {
  return code == ReasonCode::Ok;
}

[[nodiscard]] constexpr bool timed_status_ok(const ReasonCode code) noexcept {
  return code == ReasonCode::Ok || code == ReasonCode::IoTimedOut;
}

[[nodiscard]] inline std::string_view
status_error(const ReasonCode code) noexcept {
  return status_ok(code) ? std::string_view{}
                         : std::string_view{ReasonString(code)};
}

[[nodiscard]] inline std::string_view
timed_status_error(const ReasonCode code) noexcept {
  return timed_status_ok(code) ? std::string_view{}
                               : std::string_view{ReasonString(code)};
}

[[nodiscard]] constexpr int status_exit(const ReasonCode code) noexcept {
  return status_ok(code) ? 0 : 1;
}

[[nodiscard]] constexpr int timed_status_exit(const ReasonCode code) noexcept {
  return timed_status_ok(code) ? 0 : 1;
}

} // namespace detail

} // namespace rund
