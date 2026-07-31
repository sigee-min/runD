#pragma once

#include <accel/api.hpp>
#include <accel/device.hpp>

#include "base.hpp"

namespace node_accel_contract::backend_pick {

[[nodiscard]] inline bool FakeWindowContracts(const rund::AccelDevice &pick) {
  TEST_ASSERT(pick.api == rund::AccelApi::Fake);
  TEST_ASSERT(pick.caps.ok);
  TEST_ASSERT(pick.caps.api == rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(std::string_view{pick.caps.reason} == "ok");
  TEST_ASSERT(
      rund::kernel::ComputeStorageAlignmentValid(pick.caps.storage_alignment));
  TEST_ASSERT(static_cast<bool>(pick.backend));
  TEST_ASSERT(pick.owner != nullptr);
  return true;
}

[[nodiscard]] inline bool FakePickContracts() {
  rund::AccelDevice pick =
      Pick({rund::AccelApi::Vulkan, rund::AccelApi::Fake}, true);
  TEST_ASSERT(pick.check.ok);
  if (pick.api == rund::AccelApi::Vulkan) {
    TEST_ASSERT(VulkanPickHasDiscoveryCaps(pick));
  } else {
    TEST_ASSERT(FakeWindowContracts(pick));
  }

  pick = Pick({rund::AccelApi::Fake}, true);
  TEST_ASSERT(pick.check.ok);
  TEST_ASSERT(FakeWindowContracts(pick));
  return true;
}

} // namespace node_accel_contract::backend_pick
