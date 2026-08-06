#pragma once

#include <accel/api.hpp>
#include <accel/device.hpp>

#include <node/accel/pick.hpp>

#include "../reduce.hpp"
#include "range.hpp"
#include "test/assert.hpp"

namespace node_accel_contract {

[[nodiscard]] bool BackendRunsSegmentedReduce(const rund::AccelDevice &pick) {
  if (!pick.check.ok) {
    return false;
  }
  const rund::kernel::u64 submits = pick.api == rund::AccelApi::Cpu ? 0u : 1u;
  const rund::kernel::u64 dispatches = pick.api == rund::AccelApi::Cpu     ? 2u
                                       : pick.api == rund::AccelApi::Metal ? 4u
                                                                           : 4u;
  const rund::kernel::u64 range_dispatches =
      pick.api == rund::AccelApi::Cpu     ? 1u
      : pick.api == rund::AccelApi::Metal ? 4u
                                          : 4u;
  TEST_ASSERT(cpu_context::segmented::ContextRunsSegmentedReduce(
      pick, pick.api, submits, dispatches));
  TEST_ASSERT(cpu_context::segmented::ContextRunsSegmentedReduceRangeDefault(
      pick, pick.api, submits, range_dispatches));
  return true;
}

[[nodiscard]] bool AvailableBackendsRunSegmentedReduce() {
  const std::array<rund::AccelApi, 2u> apis{rund::AccelApi::Metal,
                                            rund::AccelApi::Vulkan};
  for (const rund::AccelApi api : apis) {
    const rund::AccelDevice pick =
        rund::node::accel::PickAccel(cpu_context::ApiPolicy(api));
    if (!pick.check.ok) {
      if (!cpu_context::PickUnavailableReasonIsPrecise(pick, api)) {
        return false;
      }
      continue;
    }
    if (pick.api != api || !BackendRunsSegmentedReduce(pick)) {
      return false;
    }
  }
  return true;
}

} // namespace node_accel_contract
