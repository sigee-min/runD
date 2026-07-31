#pragma once

#include <math64/nonlinear/log.hpp>
#include <math64/soa/span.hpp>

namespace rund::math64::prob {
inline constexpr i64 ProbLog2E_Q59 = 831657068615270144ll;
inline constexpr i64 ProbLn2_Q59 = 399572145162582976ll;
inline constexpr u64 ProbLogitFractionBits = 59u;
struct MaxResult {
  simd::I64x value{};
  simd::U64x index{};
  u64 processed = 0u;
  bool empty_input = true;
  [[nodiscard]] constexpr bool ok() const { return !empty_input; }
};
struct RowStatus {
  u64 processed = 0u;
  bool size_match = true;
  bool empty_input = false;
  bool overlap_ok = true;
  bool zero_sum = false;
  [[nodiscard]] constexpr bool ok() const { return size_match && overlap_ok; }
};
struct LogSumExpResult {
  simd::I64x value{};
  u64 processed = 0u;
  bool empty_input = true;
  bool zero_sum = false;
  bool saturated = false;
  [[nodiscard]] constexpr bool ok() const { return !empty_input; }
};
struct CrossEntropyResult {
  simd::I64x value{};
  u64 processed = 0u;
  bool empty_input = true;
  bool valid_target = false;
  bool zero_sum = false;
  bool saturated = false;
  [[nodiscard]] constexpr bool ok() const {
    return !empty_input && valid_target;
  }
};
} // namespace rund::math64::prob
