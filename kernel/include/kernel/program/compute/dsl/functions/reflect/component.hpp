#pragma once

namespace rund::compute_dsl::detail {

[[nodiscard]] inline ComputeValue ReflectComponent(
    const ComputeValue value, const ComputeValue projected) noexcept {
  return sub_sat(add_sat(projected, projected), value);
}

} // namespace rund::compute_dsl::detail
