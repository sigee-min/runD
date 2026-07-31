#pragma once

namespace rund::compute_dsl {

struct DifferenceSecond final {};
struct DifferenceThird final {};

struct DifferenceOrder final {
  inline static constexpr DifferenceSecond Second{};
  inline static constexpr DifferenceThird Third{};
};

[[nodiscard]] inline ComputeValue diff(const ComputeValue from,
                                       const ComputeValue to) noexcept {
  return sub_sat(to, from);
}

[[nodiscard]] inline ComputeValue diff(const ComputeValue prev,
                                       const ComputeValue center,
                                       const ComputeValue next) noexcept {
  return mean(diff(prev, center), diff(center, next));
}

[[nodiscard]] inline ComputeValue
diff(const DifferenceSecond, const ComputeValue prev, const ComputeValue center,
     const ComputeValue next) noexcept {
  return sub_sat(diff(center, next), diff(prev, center));
}

[[nodiscard]] inline ComputeValue
diff(const DifferenceThird, const ComputeValue a, const ComputeValue b,
     const ComputeValue c, const ComputeValue d) noexcept {
  return sub_sat(diff(DifferenceOrder::Second, b, c, d),
                 diff(DifferenceOrder::Second, a, b, c));
}

} // namespace rund::compute_dsl
