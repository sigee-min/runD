#include <rund/compute.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace {

template <class L, class R>
concept Addable = requires(L left, R right) { left + right; };

template <class L, class R>
concept StaticMinimum = requires(L left, R right) {
  rund::compute::min(left, right);
};

template <class L, class R>
concept StaticMaximum = requires(L left, R right) {
  rund::compute::max(left, right);
};

template <class Value>
concept FloatingRawFactory = requires(Value value) {
  rund::compute::Fixed<16, 16>::from_raw(value);
};

template <class Numerator, class Denominator>
concept FloatingRatioFactory = requires(Numerator numerator,
                                        Denominator denominator) {
  rund::compute::Fixed<16, 16>::from_ratio(numerator, denominator);
};

template <unsigned IntegerBits, unsigned FractionBits>
concept StoredFixed = requires {
  typename rund::compute::Fixed<IntegerBits, FractionBits>;
};

using StaticFormatA = rund::compute::detail::StaticArgT<
    rund::compute::Expr<rund::compute::Fixed<16, 16>>, 0u>;
using StaticFormatB = rund::compute::detail::StaticArgT<
    rund::compute::Expr<rund::compute::Fixed<8, 24>>, 1u>;

static_assert(sizeof(rund::compute::Fixed<16, 16>) == 4u);
static_assert(sizeof(rund::compute::Fixed<32, 32>) == 8u);
static_assert(alignof(rund::compute::Fixed<16, 16>) == alignof(std::int32_t));
static_assert(alignof(rund::compute::Fixed<32, 32>) == alignof(std::int64_t));
static_assert(std::is_standard_layout_v<rund::compute::Fixed<16, 16>>);
static_assert(std::is_standard_layout_v<rund::compute::Fixed<32, 32>>);
static_assert(std::is_trivially_copyable_v<rund::compute::Fixed<16, 16>>);
static_assert(std::is_trivially_copyable_v<rund::compute::Fixed<32, 32>>);
static_assert(!StoredFixed<0u, 32u>);
static_assert(!StoredFixed<32u, 0u>);
static_assert(!StoredFixed<16u, 15u>);
static_assert(!StoredFixed<64u, 64u>);
static_assert(!StoredFixed<std::numeric_limits<unsigned>::max(), 33u>);
static_assert(!std::is_convertible_v<float, rund::compute::Fixed<16, 16>>);
static_assert(!std::is_convertible_v<double, rund::compute::Fixed<32, 32>>);
static_assert(!std::is_convertible_v<rund::compute::Fixed<16, 16>, float>);
static_assert(!std::is_convertible_v<rund::compute::Fixed<32, 32>, double>);
static_assert(!FloatingRawFactory<float>);
static_assert(!FloatingRawFactory<double>);
static_assert(!FloatingRawFactory<long double>);
static_assert(!FloatingRatioFactory<float, std::int64_t>);
static_assert(!FloatingRatioFactory<std::int64_t, double>);
static_assert(!FloatingRatioFactory<long double, long double>);
static_assert(!Addable<rund::compute::Fixed<16, 16>,
                       rund::compute::Fixed<8, 24>>);
static_assert(!StaticMinimum<StaticFormatA, StaticFormatB>);
static_assert(!StaticMaximum<StaticFormatA, StaticFormatB>);

} // namespace

int RunComputeFixedValueContract() {
  using rund::compute::Fixed;
  using rund::compute::Rounding;

  const auto half32 = Fixed<1, 31>::from_ratio(1, 2, Rounding::NearestEven);
  const auto half64 = Fixed<1, 63>::from_ratio(1, 2, Rounding::NearestEven);
  const auto invalid = Fixed<1, 63>::from_ratio(1, 0, Rounding::TowardZero);
  if (!half32 || half32->raw() != 0x40000000 || !half64 ||
      half64->raw() != 0x4000000000000000ll || invalid ||
      invalid.error() != "compute_fixed_ratio_zero_denominator") {
    return 1;
  }
  if (Fixed<1, 31>::zero().raw() != 0 ||
      Fixed<1, 31>::min().raw() != std::numeric_limits<std::int32_t>::min() ||
      Fixed<1, 31>::max().raw() != std::numeric_limits<std::int32_t>::max() ||
      Fixed<1, 63>::min() >= Fixed<1, 63>::zero() || Fixed<1, 63>::max() <= Fixed<1, 63>::zero()) {
    return 2;
  }
  constexpr Fixed<1, 31> quarter = Fixed<1, 31>::ratio<1, 4, Rounding::NearestEven>();
  constexpr Fixed<1, 63> negative_half =
      Fixed<1, 63>::ratio<-1, 2, Rounding::NearestEven>();
  static_assert(quarter.raw() == 0x20000000);
  static_assert(negative_half.raw() == -0x4000000000000000ll);
  const auto toward = Fixed<1, 31>::from_ratio(1, 3, Rounding::TowardZero);
  const auto down = Fixed<1, 31>::from_ratio(1, 3, Rounding::Down);
  const auto up = Fixed<1, 31>::from_ratio(1, 3, Rounding::Up);
  const auto nearest = Fixed<1, 31>::from_ratio(1, 3, Rounding::NearestEven);
  const auto negative_down = Fixed<1, 31>::from_ratio(-1, 3, Rounding::Down);
  const auto negative_up = Fixed<1, 31>::from_ratio(-1, 3, Rounding::Up);
  if (!toward || !down || !up || !nearest || !negative_down || !negative_up ||
      toward->raw() != 715827882 || down->raw() != 715827882 ||
      up->raw() != 715827883 || nearest->raw() != 715827883 ||
      negative_down->raw() != -715827883 ||
      negative_up->raw() != -715827882) {
    return 3;
  }
  const auto high = Fixed<1, 63>::from_ratio(2, 1);
  const auto low = Fixed<1, 63>::from_ratio(-2, 1);
  if (!high || !low || high->raw() != std::numeric_limits<std::int64_t>::max() ||
      low->raw() != std::numeric_limits<std::int64_t>::min()) {
    return 4;
  }
  constexpr auto tie_even =
      Fixed<1, 31>::ratio<1, 4294967296ll, Rounding::NearestEven>();
  constexpr auto tie_odd =
      Fixed<1, 31>::ratio<3, 4294967296ll, Rounding::NearestEven>();
  static_assert(tie_even.raw() == 0);
  static_assert(tie_odd.raw() == 2);
  using Source = Fixed<16, 16>;
  using HalfFraction = Fixed<17, 15>;
  constexpr auto rounded_even =
      rund::compute::quantize<HalfFraction>(Source::from_raw(1));
  constexpr auto rounded_odd =
      rund::compute::quantize<HalfFraction>(Source::from_raw(3));
  constexpr auto rounded_negative_even =
      rund::compute::quantize<HalfFraction>(Source::from_raw(-1));
  constexpr auto rounded_negative_odd =
      rund::compute::quantize<HalfFraction>(Source::from_raw(-3));
  constexpr auto rounded_toward =
      rund::compute::quantize<HalfFraction, Rounding::TowardZero>(
          Source::from_raw(-1));
  constexpr auto rounded_down =
      rund::compute::quantize<HalfFraction, Rounding::Down>(
          Source::from_raw(-1));
  constexpr auto rounded_up =
      rund::compute::quantize<HalfFraction, Rounding::Up>(
          Source::from_raw(1));
  static_assert(rounded_even.raw() == 0);
  static_assert(rounded_odd.raw() == 2);
  static_assert(rounded_negative_even.raw() == 0);
  static_assert(rounded_negative_odd.raw() == -2);
  static_assert(rounded_toward.raw() == 0);
  static_assert(rounded_down.raw() == -1);
  static_assert(rounded_up.raw() == 1);

  using Narrow = Fixed<16, 16>;
  constexpr auto saturated_high =
      rund::compute::quantize<Narrow>(Fixed<32, 32>::max());
  constexpr auto saturated_low =
      rund::compute::quantize<Narrow>(Fixed<32, 32>::min());
  constexpr auto wrapped_32 =
      rund::compute::quantize<Narrow, Rounding::NearestEven,
                              rund::compute::Overflow::Wrap>(
          Fixed<32, 32>::max());
  constexpr auto wrapped_64 =
      rund::compute::quantize<Fixed<1, 63>, Rounding::NearestEven,
                              rund::compute::Overflow::Wrap>(
          Fixed<32, 32>::max());
  static_assert(saturated_high.raw() == std::numeric_limits<std::int32_t>::max());
  static_assert(saturated_low.raw() == std::numeric_limits<std::int32_t>::min());
  static_assert(wrapped_32.raw() == 0);
  static_assert(wrapped_64.raw() == -2147483648ll);
  return 0;
}
