#pragma once

#include <math32/nonlinear/log.hpp>

namespace rund::math32::nn {
struct RowStatus {
  u32 processed = 0u;
  bool size_match = true;
  bool empty_input = false;
  bool overlap_ok = true;
  bool valid_epsilon = true;
  bool even_length = true;
  [[nodiscard]] constexpr bool ok() const {
    return size_match && overlap_ok && valid_epsilon && even_length;
  }
};
struct RopePairResult {
  simd::I32x even{};
  simd::I32x odd{};
};
} // namespace rund::math32::nn
