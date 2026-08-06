#pragma once

#include <accel/api.hpp>
#include <accel/device.hpp>

#include <node/accel/pick.hpp>

#include "../run.hpp"

namespace node_accel_contract {

[[nodiscard]] bool BackendRunsSegmentedScan(const rund::AccelDevice &pick) {
  if (!pick.check.ok) {
    return false;
  }
  rund::kernel::u64 submits = 1u;
  rund::kernel::u64 dispatches = 4u;
  if (pick.api == rund::AccelApi::Cpu) {
    submits = 0u;
    dispatches = 3u;
  }
  return cpu_context::segmented::ContextRunsSegmentedScanThenMap(
      pick, pick.api, submits, dispatches);
}

[[nodiscard]] bool AvailableBackendsRunSegmentedScan() {
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
    if (pick.api != api || !BackendRunsSegmentedScan(pick)) {
      return false;
    }
  }
  return true;
}

} // namespace node_accel_contract
