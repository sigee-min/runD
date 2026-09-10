#pragma once

#include "../../../range_aggregate/execution/projection.hpp"

#include <string_view>

namespace rund::node::accel::detail {

[[nodiscard]] inline const char *MetalRangeOpName(const RangeOp op) noexcept {
  if (op == RangeOp::Minimum) {
    return "min";
  }
  if (op == RangeOp::Maximum) {
    return "max";
  }
  return "sum";
}

[[nodiscard]] inline const char *
MetalRangeIdentity(const RangeOp op, const std::string_view suffix) noexcept {
  if (op == RangeOp::Minimum) {
    if (suffix == "u32") {
      return "0xffffffffu";
    }
    if (suffix == "u64") {
      return "0xfffffffffffffffful";
    }
    if (suffix == "i32") {
      return "2147483647";
    }
    return "9223372036854775807l";
  }
  if (suffix == "i32") {
    return "(-2147483647 - 1)";
  }
  if (suffix == "i64") {
    return "(-9223372036854775807l - 1l)";
  }
  return suffix == "u64" ? "0ul" : "0u";
}

[[nodiscard]] inline const char *
MetalRangeUpdate(const RangeOp op, const bool saturating) noexcept {
  if (op == RangeOp::Minimum) {
    return "      value = min(value, sample);\n";
  }
  if (op == RangeOp::Maximum) {
    return "      value = max(value, sample);\n";
  }
  return saturating ? "      value = rund_range_add_sat(value, sample);\n"
                    : "      value += sample;\n";
}

template <typename Sink>
inline void AppendMetalRangeSaturatingAlgebra(Sink &source) {
  source +=
      R"MSL(inline int rund_range_add_sat(const int left, const int right) {
  if (right > 0 && left > 2147483647 - right) { return 2147483647; }
  if (right < 0 && left < (-2147483647 - 1) - right) {
    return (-2147483647 - 1);
  }
  return left + right;
}

inline long rund_range_add_sat(const long left, const long right) {
  if (right > 0l && left > 9223372036854775807l - right) {
    return 9223372036854775807l;
  }
  if (right < 0l && left < (-9223372036854775807l - 1l) - right) {
    return (-9223372036854775807l - 1l);
  }
  return left + right;
}

)MSL";
}

} // namespace rund::node::accel::detail
