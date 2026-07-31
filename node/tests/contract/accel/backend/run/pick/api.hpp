#pragma once

#include <accel/api.hpp>
#include <accel/device.hpp>

#include <node/accel/pick.hpp>

#include "../local.hpp"

namespace node_accel_contract::backend_pick {

[[nodiscard]] inline rund::AccelDevice
Pick(const std::initializer_list<rund::AccelApi> apis,
     const bool fake = false) {
  return rund::node::accel::PickAccel(backend::Policy(apis, fake));
}

} // namespace node_accel_contract::backend_pick
