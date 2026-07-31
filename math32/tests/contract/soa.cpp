#include <math32/math32.hpp>
#include <math32/soa/range.hpp>
#include "test/assert.hpp"

#include <array>
#include <span>

namespace {
struct EngineVec3Soa32 {
  std::array<rund::math32::i32, 5> x;
  std::array<rund::math32::i32, 5> y;
  std::array<rund::math32::i32, 5> z;
};

rund::math32::geom::Vec3View View(const EngineVec3Soa32& storage) {
  return rund::math32::geom::Vec3View{.x = std::span<const rund::math32::i32>(storage.x),
                                      .y = std::span<const rund::math32::i32>(storage.y),
                                      .z = std::span<const rund::math32::i32>(storage.z)};
}

rund::math32::geom::Vec3MutView MutView(EngineVec3Soa32& storage) {
  return rund::math32::geom::Vec3MutView{.x = std::span<rund::math32::i32>(storage.x),
                                         .y = std::span<rund::math32::i32>(storage.y),
                                         .z = std::span<rund::math32::i32>(storage.z)};
}

void CheckRangeLaw() {
  using rund::math32::i32;
  using rund::math32::soa::detail::Overlaps;
  using rund::math32::soa::detail::PartiallyOverlaps;
  using rund::math32::soa::detail::SameRange;

  std::array<i32, 8u> storage{};
  const std::span<const i32> empty{storage.data(), 0u};
  const std::span<i32> same_empty{storage.data(), 0u};
  const std::span<i32> other_empty{storage.data() + 1u, 0u};
  const std::span<i32> all{storage};
  TEST_ASSERT(SameRange(empty, same_empty));
  TEST_ASSERT(!SameRange(empty, other_empty));
  TEST_ASSERT(!Overlaps(empty, all));

  const std::span<const i32> left{storage.data(), 2u};
  const std::span<i32> adjacent{storage.data() + 2u, 2u};
  TEST_ASSERT(!Overlaps(left, adjacent));

  const std::span<const i32> exact{storage.data() + 1u, 3u};
  const std::span<i32> exact_mut{storage.data() + 1u, 3u};
  TEST_ASSERT(SameRange(exact, exact_mut));
  TEST_ASSERT(Overlaps(exact, exact_mut));
  TEST_ASSERT(!PartiallyOverlaps(exact, exact_mut));

  const std::span<const i32> partial{storage.data(), 4u};
  const std::span<i32> partial_mut{storage.data() + 3u, 3u};
  TEST_ASSERT(PartiallyOverlaps(partial, partial_mut));

  const std::span<const i32> outer{storage.data(), 6u};
  const std::span<i32> inner{storage.data() + 2u, 2u};
  TEST_ASSERT(Overlaps(outer, inner));
  TEST_ASSERT(PartiallyOverlaps(outer, inner));
}
}  // namespace

int RunMath32SoaContract() {
  using namespace rund::math32;

  CheckRangeLaw();
  constexpr soa::Status pending{};
  static_assert(!pending.ok());
  static_assert(pending.reason == soa::StatusReason::NotEvaluated);
  EngineVec3Soa32 lhs{.x = {1, 2, 3, 4, 5}, .y = {10, 20, 30, 40, 50}, .z = {100, 200, 300, 400, 500}};
  EngineVec3Soa32 rhs{.x = {9, 8, 7, 6, 5}, .y = {-1, -2, -3, -4, -5}, .z = {1, 2, 3, 4, 5}};
  EngineVec3Soa32 out{};

  soa::Status status = soa::Add(View(lhs), View(rhs), MutView(out));
  TEST_ASSERT(status.ok());
  TEST_ASSERT(status.processed == 5u);
  TEST_ASSERT(out.x[0] == 10 && out.x[4] == 10);
  TEST_ASSERT(out.y[1] == 18 && out.z[4] == 505);

  status = soa::Sub(View(lhs), View(rhs), MutView(out));
  TEST_ASSERT(status.ok());
  TEST_ASSERT(out.x[2] == -4 && out.y[3] == 44 && out.z[0] == 99);

  EngineVec3Soa32 lower{.x = {0, 0, 0, 0, 0}, .y = {0, 0, 0, 0, 0}, .z = {0, 0, 0, 0, 0}};
  EngineVec3Soa32 upper{.x = {3, 3, 3, 3, 3}, .y = {35, 35, 35, 35, 35}, .z = {250, 250, 250, 250, 250}};
  status = soa::Clamp(View(lhs), View(lower), View(upper), MutView(out));
  TEST_ASSERT(status.ok());
  TEST_ASSERT(out.x[0] == 1 && out.x[4] == 3);
  TEST_ASSERT(out.y[2] == 30 && out.y[4] == 35);
  TEST_ASSERT(out.z[1] == 200 && out.z[4] == 250);

  status = soa::AddInPlace(MutView(lhs), View(rhs));
  TEST_ASSERT(status.ok());
  TEST_ASSERT(lhs.x[0] == 10 && lhs.y[4] == 45 && lhs.z[4] == 505);

  const geom::Vec3View bad_size{.x = std::span<const i32>(rhs.x.data(), 4u), .y = rhs.y, .z = rhs.z};
  status = soa::Add(bad_size, View(rhs), MutView(out));
  TEST_ASSERT(!status.ok());
  TEST_ASSERT(status.reason == soa::StatusReason::SizeMismatch);

  const geom::Vec3MutView bad_components{.x = out.x, .y = std::span<i32>(out.x.data() + 1, 4u), .z = out.z};
  status = soa::Add(View(rhs), View(rhs), bad_components);
  TEST_ASSERT(!status.ok());
  TEST_ASSERT(status.reason == soa::StatusReason::SizeMismatch || status.reason == soa::StatusReason::ComponentOverlap);

  status = soa::Add(View(rhs), View(out), MutView(out));
  TEST_ASSERT(!status.ok());
  TEST_ASSERT(status.reason == soa::StatusReason::InputOutputOverlap);
  return 0;
}
