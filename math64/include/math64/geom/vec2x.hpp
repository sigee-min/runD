#pragma once

#include <math64/simd/model.hpp>

namespace rund::math64::geom {

struct Vec2x {
  simd::I64x x;
  simd::I64x y;
};

struct Mask2x {
  simd::Mask64x x;
  simd::Mask64x y;
};

}  // namespace rund::math64::geom
