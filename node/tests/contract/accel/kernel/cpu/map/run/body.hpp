#pragma once

#include <accel/device.hpp>

#include "evidence.hpp"
#include "reject.hpp"

namespace node_accel_contract::cpu_context {

[[nodiscard]] inline bool
CpuContextRejectsInvalidMapRun(const rund::AccelDevice &pick) {
  MapRunResources resources = MakeMapRunResources(pick);
  return resources.ok && RejectsForgedMapKernel(resources) &&
         RejectsMapBindingOrder(resources);
}

} // namespace node_accel_contract::cpu_context
