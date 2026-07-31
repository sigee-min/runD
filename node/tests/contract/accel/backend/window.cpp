#include <accel/api.hpp>
#include <accel/device.hpp>

#include "run/local.hpp"
#include <node/accel/pick.hpp>

#include <iostream>

namespace {

using rund::AccelDevice;

[[nodiscard]] rund::AccelDevice Pick(const rund::AccelApi api,
                                     const bool allow_fake = false) {
  return rund::node::accel::PickAccel(
      node_accel_contract::backend::Policy({api}, allow_fake));
}

[[nodiscard]] bool BackendWindowContracts(const rund::AccelDevice &pick) {
  if (!pick.check.ok) {
    return pick.api == rund::AccelApi::Metal
               ? node_accel_contract::MetalFailsClosed(pick)
               : node_accel_contract::vulkan::FailureReasonIsPrecise(pick);
  }
  TEST_ASSERT(node_accel_contract::BackendAcceptsSimpleWindow(pick));
  TEST_ASSERT(
      node_accel_contract::BackendRejectsPlanObligationMismatches(pick));
  TEST_ASSERT(node_accel_contract::BackendRejectsPlanBeyondFrozenCaps(pick));
  TEST_ASSERT(node_accel_contract::BackendRejectsUnderreportedStaging(pick));
  return true;
}

[[nodiscard]] bool
BackendResidentWindowContract(const rund::AccelDevice &pick) {
  if (!pick.check.ok) {
    return pick.api == rund::AccelApi::Metal
               ? node_accel_contract::MetalFailsClosed(pick)
               : node_accel_contract::vulkan::FailureReasonIsPrecise(pick);
  }
  if (!node_accel_contract::ResidentWindowCollapseContract(pick)) {
    std::cerr << "resident window contract failed for api="
              << static_cast<unsigned>(pick.api) << '\n';
    return false;
  }
  return true;
}

} // namespace

int RunAccelBackendWindowContract() {
  const rund::AccelDevice fake = Pick(rund::AccelApi::Fake, true);
  TEST_ASSERT(fake.check.ok);
  TEST_ASSERT(fake.api == rund::AccelApi::Fake);
  TEST_ASSERT(BackendWindowContracts(fake));
  const rund::AccelDevice metal = Pick(rund::AccelApi::Metal);
  const rund::AccelDevice vulkan = Pick(rund::AccelApi::Vulkan);
  TEST_ASSERT(BackendWindowContracts(metal));
  TEST_ASSERT(BackendResidentWindowContract(metal));
  TEST_ASSERT(BackendWindowContracts(vulkan));
  TEST_ASSERT(BackendResidentWindowContract(vulkan));
  return 0;
}
