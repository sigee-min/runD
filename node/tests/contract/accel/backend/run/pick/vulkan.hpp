#pragma once

#include <accel/api.hpp>
#include <accel/device.hpp>

#include "metal.hpp"

namespace node_accel_contract::backend_pick {

[[nodiscard]] inline bool AutoAndVulkanContracts() {
  const rund::AccelDevice pick = Pick({rund::AccelApi::Auto}, true);
  TEST_ASSERT(pick.check.ok);
  TEST_ASSERT(pick.api == rund::AccelApi::Metal ||
              pick.api == rund::AccelApi::Vulkan ||
              pick.api == rund::AccelApi::Fake);
  TEST_ASSERT(pick.api != rund::AccelApi::Fake ||
              pick.caps.api == rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(pick.api != rund::AccelApi::Vulkan ||
              VulkanPickHasDiscoveryCaps(pick));
  TEST_ASSERT(
      pick.api != rund::AccelApi::Vulkan ||
      rund::kernel::ComputeStorageAlignmentValid(pick.caps.storage_alignment));

  const rund::AccelDevice required_vulkan = Pick({rund::AccelApi::Vulkan});
  TEST_ASSERT(RequiredVulkanPickIsPlatformAware(required_vulkan));
  TEST_ASSERT(!required_vulkan.check.ok ||
              rund::kernel::ComputeStorageAlignmentValid(
                  required_vulkan.caps.storage_alignment));
  TEST_ASSERT(VulkanDirectBackendLastErrorIsPrecise());
  return true;
}

} // namespace node_accel_contract::backend_pick
