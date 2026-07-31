#include <math64/math64.hpp>
#include <math64/soa/range.hpp>
#include "test/assert.hpp"

#include <array>
#include <span>

namespace {
struct EngineVec3Soa64 {
  std::array<rund::math64::i64, 3> x;
  std::array<rund::math64::i64, 3> y;
  std::array<rund::math64::i64, 3> z;
};

rund::math64::geom::Vec3View View(const EngineVec3Soa64& storage) {
  return rund::math64::geom::Vec3View{.x = std::span<const rund::math64::i64>(storage.x),
                                      .y = std::span<const rund::math64::i64>(storage.y),
                                      .z = std::span<const rund::math64::i64>(storage.z)};
}

rund::math64::geom::Vec3MutView MutView(EngineVec3Soa64& storage) {
  return rund::math64::geom::Vec3MutView{.x = std::span<rund::math64::i64>(storage.x),
                                         .y = std::span<rund::math64::i64>(storage.y),
                                         .z = std::span<rund::math64::i64>(storage.z)};
}

void CheckRangeLaw() {
  using rund::math64::i64;
  using rund::math64::soa::detail::Overlaps;
  using rund::math64::soa::detail::PartiallyOverlaps;
  using rund::math64::soa::detail::SameRange;

  std::array<i64, 8u> storage{};
  const std::span<const i64> empty{storage.data(), 0u};
  const std::span<i64> same_empty{storage.data(), 0u};
  const std::span<i64> other_empty{storage.data() + 1u, 0u};
  const std::span<i64> all{storage};
  TEST_ASSERT(SameRange(empty, same_empty));
  TEST_ASSERT(!SameRange(empty, other_empty));
  TEST_ASSERT(!Overlaps(empty, all));

  const std::span<const i64> left{storage.data(), 2u};
  const std::span<i64> adjacent{storage.data() + 2u, 2u};
  TEST_ASSERT(!Overlaps(left, adjacent));

  const std::span<const i64> exact{storage.data() + 1u, 3u};
  const std::span<i64> exact_mut{storage.data() + 1u, 3u};
  TEST_ASSERT(SameRange(exact, exact_mut));
  TEST_ASSERT(Overlaps(exact, exact_mut));
  TEST_ASSERT(!PartiallyOverlaps(exact, exact_mut));

  const std::span<const i64> partial{storage.data(), 4u};
  const std::span<i64> partial_mut{storage.data() + 3u, 3u};
  TEST_ASSERT(PartiallyOverlaps(partial, partial_mut));

  const std::span<const i64> outer{storage.data(), 6u};
  const std::span<i64> inner{storage.data() + 2u, 2u};
  TEST_ASSERT(Overlaps(outer, inner));
  TEST_ASSERT(PartiallyOverlaps(outer, inner));
}
}  // namespace

int RunMath64SoaContract() {
  using namespace rund::math64;

  CheckRangeLaw();
  constexpr soa::Status pending{};
  static_assert(!pending.ok());
  static_assert(pending.reason == soa::StatusReason::NotEvaluated);
  EngineVec3Soa64 lhs{.x = {1, 2, 3}, .y = {10, 20, 30}, .z = {100, 200, 300}};
  EngineVec3Soa64 rhs{.x = {9, 8, 7}, .y = {-1, -2, -3}, .z = {1, 2, 3}};
  EngineVec3Soa64 out{};

  soa::Status status = soa::Add(View(lhs), View(rhs), MutView(out));
  TEST_ASSERT(status.ok());
  TEST_ASSERT(status.processed == 3u);
  TEST_ASSERT(out.x[0] == 10 && out.x[2] == 10);
  TEST_ASSERT(out.y[1] == 18 && out.z[2] == 303);

  status = soa::Sub(View(lhs), View(rhs), MutView(out));
  TEST_ASSERT(status.ok());
  TEST_ASSERT(out.x[2] == -4 && out.y[2] == 33 && out.z[0] == 99);

  EngineVec3Soa64 lower{.x = {0, 0, 0}, .y = {0, 0, 0}, .z = {0, 0, 0}};
  EngineVec3Soa64 upper{.x = {2, 2, 2}, .y = {25, 25, 25}, .z = {250, 250, 250}};
  status = soa::Clamp(View(lhs), View(lower), View(upper), MutView(out));
  TEST_ASSERT(status.ok());
  TEST_ASSERT(out.x[0] == 1 && out.x[2] == 2);
  TEST_ASSERT(out.y[1] == 20 && out.y[2] == 25);
  TEST_ASSERT(out.z[1] == 200 && out.z[2] == 250);

  status = soa::AddInPlace(MutView(lhs), View(rhs));
  TEST_ASSERT(status.ok());
  TEST_ASSERT(lhs.x[0] == 10 && lhs.y[2] == 27 && lhs.z[2] == 303);

  const geom::Vec3View bad_size{.x = std::span<const i64>(rhs.x.data(), 2u), .y = rhs.y, .z = rhs.z};
  status = soa::Add(bad_size, View(rhs), MutView(out));
  TEST_ASSERT(!status.ok());
  TEST_ASSERT(status.reason == soa::StatusReason::SizeMismatch);

  status = soa::Add(View(rhs), View(out), MutView(out));
  TEST_ASSERT(!status.ok());
  TEST_ASSERT(status.reason == soa::StatusReason::InputOutputOverlap);
  return 0;
}
