#pragma once

#include "test/assert.hpp"
#include "test/compute/fixed.hpp"

#include <kernel/program/compute/factor/identity.hpp>
#include <kernel/program/compute/factor/model.hpp>
#include <kernel/program/compute/factor/plan.hpp>
#include <kernel/program/compute/factor/reference.hpp>
#include <kernel/program/compute/matrix/identity.hpp>
#include <kernel/program/compute/matrix/model.hpp>
#include <kernel/program/compute/matrix/plan.hpp>
#include <kernel/program/compute/matrix/reference.hpp>
#include <kernel/program/compute/solve/identity.hpp>
#include <kernel/program/compute/solve/model.hpp>
#include <kernel/program/compute/solve/plan.hpp>
#include <kernel/program/compute/solve/reference.hpp>
#include <kernel/program/compute/spectrum/identity.hpp>
#include <kernel/program/compute/spectrum/model.hpp>
#include <kernel/program/compute/spectrum/plan.hpp>
#include <kernel/program/compute/spectrum/reference.hpp>
#include <kernel/program/compute/transform/identity.hpp>
#include <kernel/program/compute/transform/model.hpp>
#include <kernel/program/compute/transform/plan.hpp>
#include <kernel/program/compute/transform/reference.hpp>

#include <array>
#include <string_view>

namespace program_compute_contract::numeric_algebra_contract {

inline constexpr rund::kernel::i32 kOne = 0x40000000;
inline constexpr rund::kernel::i32 kHalf = 0x20000000;
inline constexpr rund::kernel::i32 kQuarter = 0x10000000;
inline constexpr rund::kernel::i32 kFixedMax = 0x7fffffff;
inline constexpr rund::kernel::i64 kWideOne = rund::kernel::i64{1} << 62u;
inline constexpr rund::kernel::ComputeFixedFormat kFixedI1F31 =
    rund::kernel::PrimitiveFixedFormat(
        test::FixedFormatForLane(rund::kernel::ComputeScalar::Lane32),
        rund::kernel::ComputeApproximation::Deterministic);
inline constexpr rund::kernel::ComputeFixedFormat kFixedI1F63 =
    rund::kernel::PrimitiveFixedFormat(
        test::FixedFormatForLane(rund::kernel::ComputeScalar::Lane64),
        rund::kernel::ComputeApproximation::Deterministic);

template <typename Hash>
[[nodiscard]] constexpr bool HashesDiffer(const Hash lhs,
                                          const Hash rhs) noexcept {
  return lhs.hi != rhs.hi || lhs.lo != rhs.lo;
}

template <typename Desc, typename HashFn>
int CheckFixedFormatIdentityAxes(const Desc &base, HashFn &&hash_fn) {
  const auto baseline = hash_fn(base);
  Desc changed = base;
  changed.fixed_format.integer_bits = 2u;
  changed.fixed_format.fraction_bits = 30u;
  TEST_ASSERT(HashesDiffer(baseline, hash_fn(changed)));
  changed = base;
  changed.fixed_format.rounding = rund::kernel::ComputeRounding::Up;
  TEST_ASSERT(HashesDiffer(baseline, hash_fn(changed)));
  changed = base;
  changed.fixed_format.overflow = rund::kernel::ComputeOverflow::Wrap;
  TEST_ASSERT(HashesDiffer(baseline, hash_fn(changed)));
  changed = base;
  changed.fixed_format.approximation =
      rund::kernel::ComputeApproximation::Exact;
  TEST_ASSERT(HashesDiffer(baseline, hash_fn(changed)));
  return 0;
}

} // namespace program_compute_contract::numeric_algebra_contract
