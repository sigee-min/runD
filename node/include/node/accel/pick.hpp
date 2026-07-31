#pragma once

#include <accel/device.hpp>


namespace rund::node::accel {

[[nodiscard]] rund::AccelDevice PickAccel(const rund::AccelPolicy& policy = {});

}  // namespace rund::node::accel
