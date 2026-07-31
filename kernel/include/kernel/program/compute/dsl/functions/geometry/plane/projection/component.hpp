#pragma once

namespace rund::compute_dsl::detail {

[[nodiscard]] inline ComputeValue
PlaneProjectionComponent(const ComputeValue point_component,
                         const ComputeValue normal_component,
                         const ComputeValue t) noexcept {
  return sub_sat(point_component, mul_fixed(normal_component, t));
}

} // namespace rund::compute_dsl::detail
