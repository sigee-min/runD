#pragma once

#include <string_view>

namespace rund::kernel::program_detail {

[[nodiscard]] inline bool IsPassReason(const char* const reason) noexcept {
  return reason != nullptr && std::string_view{reason} == "pass";
}

[[nodiscard]] inline bool IsObservedReason(const char* const reason) noexcept {
  return reason != nullptr && std::string_view{reason} != "not_run";
}

} // namespace rund::kernel::program_detail
