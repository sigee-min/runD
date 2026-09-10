#include "local.hpp"

#include "test/assert.hpp"

#include <kernel/program/compute/ir.hpp>

#include <array>
#include <cstdint>
#include <limits>

namespace program_compute_contract::helper_emission {
namespace {

using NativeU128 = rund::kernel::u128;

struct OracleU128 {
  std::uint64_t hi;
  std::uint64_t lo;
};

[[nodiscard]] constexpr OracleU128 PairOf(const NativeU128 value) noexcept {
  return OracleU128{.hi = static_cast<std::uint64_t>(value >> 64u),
                    .lo = static_cast<std::uint64_t>(value)};
}

[[nodiscard]] constexpr bool OracleGe(const OracleU128 lhs,
                                      const OracleU128 rhs) noexcept {
  return lhs.hi > rhs.hi || (lhs.hi == rhs.hi && lhs.lo >= rhs.lo);
}

[[nodiscard]] constexpr OracleU128 OracleSub(const OracleU128 lhs,
                                             const OracleU128 rhs) noexcept {
  const std::uint64_t borrow = lhs.lo < rhs.lo ? 1u : 0u;
  return OracleU128{.hi = lhs.hi - rhs.hi - borrow, .lo = lhs.lo - rhs.lo};
}

[[nodiscard]] constexpr OracleU128 OracleShl1Or(const OracleU128 value,
                                                const std::uint64_t bit) {
  return OracleU128{.hi = (value.hi << 1u) | (value.lo >> 63u),
                    .lo = (value.lo << 1u) | bit};
}

[[nodiscard]] constexpr std::uint64_t
RestoringDivU128ByU64(const OracleU128 numerator,
                      const std::uint64_t denominator) noexcept {
  constexpr std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  if (denominator == 0u) {
    return max;
  }
  const OracleU128 denominator_wide{.hi = 0u, .lo = denominator};
  OracleU128 remainder{};
  std::uint64_t quotient = 0u;
  bool overflow = false;
  for (int bit = 127; bit >= 0; --bit) {
    const std::uint64_t input_bit =
        bit >= 64 ? ((numerator.hi >> static_cast<unsigned>(bit - 64)) & 1u)
                  : ((numerator.lo >> static_cast<unsigned>(bit)) & 1u);
    remainder = OracleShl1Or(remainder, input_bit);
    if (OracleGe(remainder, denominator_wide)) {
      remainder = OracleSub(remainder, denominator_wide);
      if (bit >= 64) {
        overflow = true;
      } else {
        quotient |= std::uint64_t{1u} << static_cast<unsigned>(bit);
      }
    }
  }
  return overflow ? max : quotient;
}

[[nodiscard]] constexpr std::uint64_t
NativeDivU128ByU64(const OracleU128 numerator,
                   const std::uint64_t denominator) noexcept {
  constexpr std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  if (denominator == 0u) {
    return max;
  }
  const NativeU128 value =
      (static_cast<NativeU128>(numerator.hi) << 64u) | numerator.lo;
  const NativeU128 quotient = value / denominator;
  return quotient > max ? max : static_cast<std::uint64_t>(quotient);
}

} // namespace

int CheckGenericU128Division() {
  constexpr std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  constexpr std::uint64_t high_denominator = max;
  constexpr std::uint64_t high_quotient = 0x8000000000000001ull;
  constexpr NativeU128 high_product =
      static_cast<NativeU128>(high_denominator) * high_quotient +
      high_denominator / 2u;
  struct DivisionCase {
    OracleU128 numerator;
    std::uint64_t denominator;
  };
  constexpr std::array<DivisionCase, 6u> cases{
      DivisionCase{.numerator = PairOf(0u), .denominator = 0u},
      DivisionCase{.numerator = PairOf(max), .denominator = 1u},
      DivisionCase{.numerator = PairOf(NativeU128{1u} << 127u),
                   .denominator = max},
      DivisionCase{.numerator = PairOf(high_product),
                   .denominator = high_denominator},
      DivisionCase{.numerator = PairOf(~NativeU128{0u}), .denominator = max},
      DivisionCase{.numerator = PairOf(~NativeU128{0u}), .denominator = 1u},
  };
  TEST_ASSERT((cases[3].numerator.hi & (std::uint64_t{1u} << 63u)) != 0u);
  for (const DivisionCase &value : cases) {
    TEST_ASSERT(RestoringDivU128ByU64(value.numerator, value.denominator) ==
                NativeDivU128ByU64(value.numerator, value.denominator));
  }
  TEST_ASSERT(RestoringDivU128ByU64(cases[3].numerator, cases[3].denominator) ==
              high_quotient);
  return 0;
}

} // namespace program_compute_contract::helper_emission
