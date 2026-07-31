#pragma once

#include <math32/simd/model.hpp>

namespace rund::math32::geom {

struct Vec2x {
  simd::I32x x;
  simd::I32x y;
};

struct Mask2x {
  simd::Mask32x x;
  simd::Mask32x y;
};

}  // namespace rund::math32::geom
