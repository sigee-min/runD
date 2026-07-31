#pragma once

#include <accel/device.hpp>

#include <string_view>

namespace node_accel_contract {

[[nodiscard]] inline bool ReasonIs(const rund::AccelDevice &pick,
                                   const std::string_view reason) {
  return std::string_view{pick.check.reason} == reason;
}

[[nodiscard]] inline bool MetalFailsClosed(const rund::AccelDevice &pick) {
  return !pick.check.ok &&
         (ReasonIs(pick, "accel_metal_unavailable") ||
          ReasonIs(pick, "accel_metal_device_unavailable") ||
          ReasonIs(pick, "accel_metal_device_name_invalid") ||
          ReasonIs(pick, "accel_metal_device_info_capacity") ||
          ReasonIs(pick, "accel_metal_queue_unavailable") ||
          ReasonIs(pick, "accel_metal_sdk_unavailable"));
}

} // namespace node_accel_contract
