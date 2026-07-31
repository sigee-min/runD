#pragma once

#include <math32/nonlinear/log.hpp>
#include <math32/soa/span.hpp>

namespace rund::math32::prob {
inline constexpr i32 ProbLog2E_Q27 = 193635251;
inline constexpr i32 ProbLn2_Q27 = 93032640;
inline constexpr u32 ProbLogitFractionBits = 27u;
struct MaxResult {
  simd::I32x value{};
  simd::U32x index{};
  u32 processed = 0u;
  bool empty_input = true;
  [[nodiscard]] constexpr bool ok() const { return !empty_input; }
};
struct RowStatus {
  u32 processed = 0u;
  bool size_match = true;
  bool empty_input = false;
  bool overlap_ok = true;
  bool zero_sum = false;
  [[nodiscard]] constexpr bool ok() const { return size_match && overlap_ok; }
};
struct LogSumExpResult {
  simd::I32x value{};
  u32 processed = 0u;
  bool empty_input = true;
  bool zero_sum = false;
  bool saturated = false;
  [[nodiscard]] constexpr bool ok() const { return !empty_input; }
};
struct CrossEntropyResult {
  simd::I32x value{};
  u32 processed = 0u;
  bool empty_input = true;
  bool valid_target = false;
  bool zero_sum = false;
  bool saturated = false;
  [[nodiscard]] constexpr bool ok() const {
    return !empty_input && valid_target;
  }
};
} // namespace rund::math32::prob
