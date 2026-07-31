#pragma once

#include <kernel/program/compute/dsl/functions/geometry/projection/scale.hpp>

namespace rund::compute_dsl::detail {

[[nodiscard]] inline ComputeValue
ProjectComponent(const ComputeValue component, const ComputeValue dot_value,
                 const ComputeValue denom) noexcept {
  return mul_fixed(component, ProjectScale(dot_value, denom));
}

} // namespace rund::compute_dsl::detail
