#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue fixed(const FixedOpHalf,
                                        const ComputeValue value) noexcept {
  return detail::FixedRatioConstant(value, 1u, 2u);
}

[[nodiscard]] inline ComputeValue fixed(const FixedOpThird,
                                        const ComputeValue value) noexcept {
  return detail::FixedRatioConstant(value, 1u, 3u);
}

[[nodiscard]] inline ComputeValue fixed(const FixedOpQuarter,
                                        const ComputeValue value) noexcept {
  return detail::FixedRatioConstant(value, 1u, 4u);
}

} // namespace rund::compute_dsl
