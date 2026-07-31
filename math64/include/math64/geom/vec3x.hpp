#pragma once

#include <math64/geom/vec2x.hpp>

namespace rund::math64::geom {

struct Vec3x {
  simd::I64x x;
  simd::I64x y;
  simd::I64x z;
};

struct Mask3x {
  simd::Mask64x x;
  simd::Mask64x y;
  simd::Mask64x z;
};

}  // namespace rund::math64::geom
