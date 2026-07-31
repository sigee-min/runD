#include "test/assert.hpp"

#include <math32/math32.hpp>

namespace {
void ExpectVec2x(const rund::math32::geom::Vec2x value,
                 const rund::math32::i32 x0,
                 const rund::math32::i32 x1,
                 const rund::math32::i32 y0,
                 const rund::math32::i32 y1) {
  alignas(16) rund::math32::i32 x[rund::math32::simd::LaneCount]{};
  alignas(16) rund::math32::i32 y[rund::math32::simd::LaneCount]{};
  rund::math32::simd::Store(x, value.x);
  rund::math32::simd::Store(y, value.y);
  TEST_ASSERT(x[0] == x0);
  TEST_ASSERT(x[1] == x1);
  TEST_ASSERT(y[0] == y0);
  TEST_ASSERT(y[1] == y1);
}

void ExpectVec3xZ01(const rund::math32::geom::Vec3x value,
                    const rund::math32::i32 z0,
                    const rund::math32::i32 z1) {
  alignas(16) rund::math32::i32 z[rund::math32::simd::LaneCount]{};
  rund::math32::simd::Store(z, value.z);
  TEST_ASSERT(z[0] == z0);
  TEST_ASSERT(z[1] == z1);
}

void ExpectMask2x(const rund::math32::geom::Mask2x value, const bool x0, const bool y1) {
  alignas(16) rund::math32::u32 x[rund::math32::simd::LaneCount]{};
  alignas(16) rund::math32::u32 y[rund::math32::simd::LaneCount]{};
  rund::math32::simd::Store(x, value.x);
  rund::math32::simd::Store(y, value.y);
  TEST_ASSERT((x[0] == rund::math32::simd::MaskTrueLane) == x0);
  TEST_ASSERT((y[1] == rund::math32::simd::MaskTrueLane) == y1);
}
}  // namespace

int RunMath32GeometryVectorContract() {
  using namespace rund::math32;
  using namespace rund::math32::geom;

  constexpr ViewStatus pending{};
  static_assert(!pending.ok());
  static_assert(!pending);
  static_assert(pending.reason == ViewStatusReason::NotEvaluated);
  const Vec2x ax{.x = simd::I32x{1, -2, 3, -4}, .y = simd::I32x{5, -6, 7, -8}};
  const Vec2x bx{.x = simd::I32x{10, 20, 30, 40}, .y = simd::I32x{-5, -6, -7, -8}};
  ExpectVec2x(Add(ax, bx), 11, 18, 0, -12);
  ExpectVec2x(Sub(ax, bx), -9, -22, 10, 0);
  ExpectVec2x(Neg(ax), -1, 2, -5, 6);
  ExpectVec2x(Abs(ax), 1, 2, 5, 6);
  ExpectVec2x(Min(ax, bx), 1, -2, -5, -6);
  ExpectVec2x(Max(ax, bx), 10, 20, 5, -6);
  ExpectVec2x(Clamp(ax, Vec2x{.x = simd::SplatI32(-1), .y = simd::SplatI32(-5)},
                    Vec2x{.x = simd::SplatI32(2), .y = simd::SplatI32(6)}),
              1,
              -1,
              5,
              -5);
  ExpectMask2x(Lt(ax, bx), true, false);
  ExpectMask2x(Le(ax, ax), true, true);
  ExpectMask2x(Eq(ax, ax), true, true);
  ExpectVec2x(Select(Lt(ax, bx), ax, bx), 1, -2, -5, -6);
  const Vec3x c3{.x = simd::SplatI32(1), .y = simd::SplatI32(2), .z = simd::I32x{3, 4, 5, 6}};
  const Vec3x d3{.x = simd::SplatI32(10), .y = simd::SplatI32(20), .z = simd::I32x{30, 40, 50, 60}};
  ExpectVec3xZ01(Add(c3, d3), 33, 44);

  i32 x[simd::LaneCount + 1u]{1, 2, 3, 4, 99};
  i32 y[simd::LaneCount + 1u]{10, 20, 30, 40, 88};
  i32 z[simd::LaneCount + 1u]{100, 200, 300, 400, 77};
  i32 out_x[simd::LaneCount + 1u]{};
  i32 out_y[simd::LaneCount + 1u]{};
  i32 out_z[simd::LaneCount + 1u]{};
  const Vec3View input{.x = std::span<const i32>(x), .y = std::span<const i32>(y), .z = std::span<const i32>(z)};
  const Vec3MutView output{.x = std::span<i32>(out_x), .y = std::span<i32>(out_y), .z = std::span<i32>(out_z)};
  TEST_ASSERT(Validate(input).ok());
  TEST_ASSERT(Validate(output).ok());
  TEST_ASSERT(CanLoad(input, 0u).ok());
  TEST_ASSERT(CanStore(output, 0u).ok());
  TEST_ASSERT(!CanLoad(input, 2u).ok());
  TEST_ASSERT(CanLoad(input, 2u).reason == ViewStatusReason::LaneOutOfRange);
  TEST_ASSERT(!CanStore(output, 2u).ok());
  TEST_ASSERT(CanStore(output, 2u).reason == ViewStatusReason::LaneOutOfRange);
  Store(output, 0u, Load(input, 0u));
  TEST_ASSERT(out_x[0] == 1 && out_y[1] == 20 && out_z[3] == 400);
  const Vec2View bad_size{.x = std::span<const i32>(x, 4u), .y = std::span<const i32>(y, 3u)};
  TEST_ASSERT(!Validate(bad_size).ok());
  TEST_ASSERT(Validate(bad_size).reason == ViewStatusReason::SizeMismatch);
  const Vec2MutView bad_overlap{.x = std::span<i32>(out_x, 4u), .y = std::span<i32>(out_x + 1, 4u)};
  TEST_ASSERT(!Validate(bad_overlap).ok());
  TEST_ASSERT(Validate(bad_overlap).reason == ViewStatusReason::ComponentOverlap);
  return 0;
}
