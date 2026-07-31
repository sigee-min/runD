#pragma once

#include <cstdint>

namespace rund {

struct AccelCheck {
  bool ok = false;
  const char* reason = "invalid";
  std::uint64_t failed_batches = 0u;
  std::uint64_t first_failed_batch = 0u;
  std::uint32_t first_status = 0u;

  constexpr AccelCheck() = default;

  constexpr AccelCheck(const bool valid, const char* const why) noexcept
      : ok(valid), reason(why) {}

  template <typename Status>
    requires requires(const Status& status) {
      status.ok;
      status.reason;
    }
  constexpr AccelCheck(const Status& status) noexcept
      : ok(status.ok), reason(status.reason) {}

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok;
  }
};

}  // namespace rund
