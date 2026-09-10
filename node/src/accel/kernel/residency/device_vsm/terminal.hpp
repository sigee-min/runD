#pragma once

#include "request.hpp"

#include <array>
#include <cstdint>

namespace rund::node::accel::detail {

struct DeviceVsmNativeExecution final {
  rund::AccelCheck accepted{false, "accel_kernel_pipeline_invalid"};
  std::array<std::uint32_t, DeviceVsmResultWordCount> counters{};
  std::uint32_t window_seed_dispatches{};
  std::uint32_t window_compute_dispatches{};
  std::uint32_t window_internal_dispatches{};
  std::uint32_t encoded_tile_count{};
  std::uint64_t completed_ns{};
  std::uint64_t kernel_ns{};
  std::uint64_t kernel_samples{};
  std::uint64_t submit_wait_ns{};
  std::uint64_t native_check_reason{};
  std::uint32_t native_check_code{};
  bool result_mapped{};
  bool native_check_ok{};
  bool result_acquired{};
};

[[nodiscard]] inline std::uint64_t
device_vsm_reason_code(const char *const text) noexcept {
  constexpr std::uint64_t Offset = 1469598103934665603ull;
  constexpr std::uint64_t Prime = 1099511628211ull;
  std::uint64_t value = Offset;
  if (text == nullptr) {
    return value;
  }
  for (const auto *cursor = reinterpret_cast<const unsigned char *>(text);
       *cursor != 0u; ++cursor) {
    value ^= *cursor;
    value *= Prime;
  }
  return value;
}

// Sole aggregate terminal authority. Backends own submission and result
// acquisition; this owner alone translates the fixed result words into the
// authenticated product receipt.
[[nodiscard]] DeviceVsmFinal
ClassifyDeviceVsmTerminal(const DeviceVsmRequest &,
                          const DeviceVsmNativeExecution &) noexcept;

} // namespace rund::node::accel::detail
