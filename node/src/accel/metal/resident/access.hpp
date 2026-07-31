#pragma once

#include "../state.hpp"
#include "state.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] inline MetalResidentState &
MetalResidents(MetalAdapter &adapter) noexcept {
  return *adapter.resident;
}

[[nodiscard]] inline const MetalResidentState &
MetalResidents(const MetalAdapter &adapter) noexcept {
  return *adapter.resident;
}

} // namespace rund::node::accel::detail
