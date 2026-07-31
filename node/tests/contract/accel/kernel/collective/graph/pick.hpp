#pragma once

#include <accel/api.hpp>
#include <accel/device.hpp>

#include <node/accel/pick.hpp>

#include "../local.hpp"

namespace node_accel_contract::collective::graph_case {

[[nodiscard]] inline rund::AccelDevice PickGraphBackend() {
  rund::AccelDevice pick =
      rund::node::accel::PickAccel(Policy(rund::AccelApi::Metal));
  if (pick.check.ok) {
    return pick;
  }
  if (!PickUnavailableReasonIsPrecise(pick, rund::AccelApi::Metal)) {
    return pick;
  }
  return rund::node::accel::PickAccel(Policy(rund::AccelApi::Vulkan));
}

} // namespace node_accel_contract::collective::graph_case
