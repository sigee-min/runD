#pragma once

#include "../local.hpp"

#include <string_view>

namespace rund::node::accel::detail {

[[nodiscard]] inline const char *
VulkanRangeScalar(const bool wide, const bool signed_values) noexcept {
  return signed_values ? (wide ? "int64_t" : "int")
                       : (wide ? "uint64_t" : "uint");
}

template <typename Sink>
[[nodiscard]] bool EmitVulkanRangeUpdate(
    Sink &sink, const RangeOp op,
    const bool saturating) noexcept(noexcept(sink.append(std::string_view{}))) {
  if (op == RangeOp::Minimum) {
    return sink.append("      value = min(value, item);\n");
  }
  if (op == RangeOp::Maximum) {
    return sink.append("      value = max(value, item);\n");
  }
  return sink.append(saturating
                         ? "      value = rund_range_add_sat(value, item);\n"
                         : "      value += item;\n");
}

template <typename Sink>
[[nodiscard]] bool EmitVulkanRangeSaturatingAlgebra(Sink &sink) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  return sink.append(
      R"glsl(int rund_range_add_sat(const int left, const int right) {
  if (right > 0 && left > 2147483647 - right) { return 2147483647; }
  if (right < 0 && left < (-2147483647 - 1) - right) {
    return (-2147483647 - 1);
  }
  return left + right;
}

int64_t rund_range_add_sat(const int64_t left, const int64_t right) {
  const int64_t maximum = int64_t(0x7fffffffffffffffUL);
  const int64_t minimum = -maximum - int64_t(1);
  if (right > int64_t(0) && left > maximum - right) { return maximum; }
  if (right < int64_t(0) && left < minimum - right) { return minimum; }
  return left + right;
}

)glsl");
}

[[nodiscard]] constexpr std::uint32_t
VulkanRangeStageValue(const RangeStageKind stage) noexcept {
  return static_cast<std::uint32_t>(stage);
}

[[nodiscard]] inline const char *VulkanRangeCombine(const RangeOp op) noexcept {
  return op == RangeOp::Minimum ? "min" : "max";
}

[[nodiscard]] inline const char *
VulkanRangeIdentity(const RangeOp op, const bool wide,
                    const bool signed_values) noexcept {
  return op == RangeOp::Minimum
             ? (signed_values
                    ? (wide ? "int64_t(0x7fffffffffffffffUL)" : "2147483647")
                    : (wide ? "uint64_t(0xffffffffffffffffUL)" : "0xffffffffu"))
             : (signed_values
                    ? (wide ? "(-int64_t(0x7fffffffffffffffUL) - int64_t(1))"
                            : "(-2147483647 - 1)")
                    : (wide ? "uint64_t(0)" : "0u"));
}

} // namespace rund::node::accel::detail
