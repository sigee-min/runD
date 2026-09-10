#pragma once

#include <accel/api.hpp>
#include <accel/device.hpp>

namespace node_accel_contract::vulkan {

[[nodiscard]] inline rund::AccelPolicy RequiredPolicy() {
  rund::AccelPolicy policy{};
  policy.preferred[0] = rund::AccelApi::Vulkan;
  policy.preferred_count = 1u;
  policy.allow_fake = false;
  return policy;
}

} // namespace node_accel_contract::vulkan
