#pragma once

#include "../../range_aggregate/model.hpp"

#include <accel/device.hpp>

namespace rund::node::accel::detail {

[[nodiscard]] RangeCaps MetalRangeCaps(const rund::AccelDevice &pick) noexcept;

} // namespace rund::node::accel::detail
