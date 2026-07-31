#pragma once

#include <math64/nonlinear/log.hpp>

namespace rund::math64::nn {
struct RowStatus {
  u64 processed = 0u;
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
  simd::I64x even{};
  simd::I64x odd{};
};
} // namespace rund::math64::nn
