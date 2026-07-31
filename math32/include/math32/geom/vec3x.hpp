#pragma once

#include <math32/geom/vec2x.hpp>

namespace rund::math32::geom {

struct Vec3x {
  simd::I32x x;
  simd::I32x y;
  simd::I32x z;
};

struct Mask3x {
  simd::Mask32x x;
  simd::Mask32x y;
  simd::Mask32x z;
};

}  // namespace rund::math32::geom
