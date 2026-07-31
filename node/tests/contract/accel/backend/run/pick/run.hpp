#pragma once

#include "vulkan.hpp"

namespace node_accel_contract {

int RunAccelPickContracts() {
  TEST_ASSERT(backend_pick::BasePickContracts());
  TEST_ASSERT(backend_pick::FakePickContracts());
  TEST_ASSERT(backend_pick::MetalIdentityContract());
  TEST_ASSERT(backend_pick::AutoAndVulkanContracts());
  return 0;
}

} // namespace node_accel_contract
