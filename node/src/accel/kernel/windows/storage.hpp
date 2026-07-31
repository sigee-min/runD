#pragma once

#include <kernel/program/compute/backend.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace rund::node::accel::detail {

struct DispatchWindowStorage {
  std::array<rund::kernel::ComputeDispatchWindow, 1u> inline_windows{};
  std::vector<rund::kernel::ComputeDispatchWindow> overflow_windows{};
  std::uint64_t window_count = 0u;
  bool ok = true;
  const char* reason = "ok";

  [[nodiscard]] const rund::kernel::ComputeDispatchWindow* data() const noexcept {
    if (window_count == 0u) {
      return nullptr;
    }
    return overflow_windows.empty() ? inline_windows.data()
                                    : overflow_windows.data();
  }

  [[nodiscard]] std::uint64_t size() const noexcept {
    return window_count;
  }
};

[[nodiscard]] inline DispatchWindowStorage RejectDispatchWindows(
    const char* const reason) noexcept {
  DispatchWindowStorage storage{};
  storage.ok = false;
  storage.reason = reason;
  return storage;
}

}  // namespace rund::node::accel::detail
