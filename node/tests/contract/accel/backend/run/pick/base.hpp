#pragma once

#include <accel/api.hpp>
#include <accel/device.hpp>

#include <node/accel/pick.hpp>

#include "api.hpp"

#include <cstdint>
#include <cstdio>
#include <iterator>

namespace node_accel_contract::backend_pick {

[[nodiscard]] inline bool BasePickContracts() {
  rund::AccelDevice pick = rund::node::accel::PickAccel();
  TEST_ASSERT(pick.api != rund::AccelApi::Fake);
  if (!pick.check.ok) {
    if (!ReasonIs(pick, "compute_adapter_unavailable") &&
        !MetalFailsClosed(pick) && !vulkan::FailureReasonIsPrecise(pick)) {
      std::fprintf(stderr, "unexpected accel pick api=%u reason=%s\n",
                   static_cast<unsigned>(pick.api), pick.check.reason);
    }
    TEST_ASSERT(ReasonIs(pick, "compute_adapter_unavailable") ||
                MetalFailsClosed(pick) || vulkan::FailureReasonIsPrecise(pick));
  }

  pick = Pick({rund::AccelApi::Auto});
  TEST_ASSERT(pick.api != rund::AccelApi::Fake);
  if (!pick.check.ok) {
    TEST_ASSERT(ReasonIs(pick, "compute_adapter_unavailable"));
  }

  pick = Pick({rund::AccelApi::Cpu});
  TEST_ASSERT(pick.api == rund::AccelApi::Cpu);
  TEST_ASSERT(pick.check.ok);
  TEST_ASSERT(ReasonIs(pick, "ok"));
  TEST_ASSERT(pick.cpu_caps.ok);
  TEST_ASSERT(pick.cpu_caps.backend == rund::kernel::ComputeBackend::Cpu);
  TEST_ASSERT(
      rund::kernel::ComputeStorageAlignmentValid(pick.caps.storage_alignment));

  pick = Pick({rund::AccelApi::Fake});
  TEST_ASSERT(!pick.check.ok);
  TEST_ASSERT(ReasonIs(pick, "accel_fake_disallowed"));

  rund::AccelPolicy invalid{};
  invalid.preferred_count =
      static_cast<std::uint32_t>(std::size(invalid.preferred) + 1u);
  pick = rund::node::accel::PickAccel(invalid);
  TEST_ASSERT(!pick.check.ok);
  TEST_ASSERT(ReasonIs(pick, "compute_policy_invalid"));
  return true;
}

} // namespace node_accel_contract::backend_pick
