#pragma once

namespace rund {

struct AccelKernelCheck {
  bool ok = false;
  const char* reason = "accel_kernel_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok;
  }
};

}  // namespace rund
