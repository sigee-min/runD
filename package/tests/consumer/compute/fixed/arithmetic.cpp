#include "local.hpp"

#include <rund/compute.hpp>
#include <rund/compute/math.hpp>

#include <array>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace package_compute::fixed_contract {

namespace {

struct FixedImpostor {
  static constexpr unsigned integer_bits = 16u;
  static constexpr unsigned fraction_bits = 16u;
  using Raw = std::int32_t;
  static constexpr FixedImpostor from_raw(Raw) noexcept { return {}; }
};

template <class Target>
concept HostQuantizable = requires(rund::compute::Fixed<16, 16> value) {
  rund::compute::quantize<Target>(value);
};

template <class Left, class Right>
concept HostAddable = requires(Left left, Right right) { left + right; };

template <class Left, class Right>
concept HostSubtractable = requires(Left left, Right right) { left - right; };

template <class Left, class Right>
concept HostMultipliable = requires(Left left, Right right) { left * right; };

template <class Left, class Right>
concept HostDividable = requires(Left left, Right right) { left / right; };

template <class Value>
concept HostFloatCastable =
    requires(Value value) { static_cast<float>(value); };

template <class Value>
concept HostDoubleCastable =
    requires(Value value) { static_cast<double>(value); };

template <class Value>
concept HostFloatingRawFactory =
    requires(Value value) { rund::compute::Fixed<16, 16>::from_raw(value); };

template <class Numerator, class Denominator>
concept HostFloatingRatioFactory =
    requires(Numerator numerator, Denominator denominator) {
      rund::compute::Fixed<16, 16>::from_ratio(numerator, denominator);
    };

static_assert(!HostQuantizable<FixedImpostor>);
static_assert(FixedImpostor::integer_bits == 16u &&
              FixedImpostor::fraction_bits == 16u);
static_assert(!std::is_convertible_v<float, rund::compute::Fixed<16, 16>>);
static_assert(!std::is_convertible_v<double, rund::compute::Fixed<32, 32>>);
static_assert(!std::is_convertible_v<rund::compute::Fixed<16, 16>, float>);
static_assert(!std::is_convertible_v<rund::compute::Fixed<32, 32>, double>);
static_assert(
    !HostAddable<rund::compute::Fixed<16, 16>, rund::compute::Fixed<17, 15>>);
static_assert(!HostSubtractable<rund::compute::Fixed<16, 16>,
                                rund::compute::Fixed<17, 15>>);
static_assert(!HostMultipliable<rund::compute::Fixed<16, 16>,
                                rund::compute::Fixed<17, 15>>);
static_assert(
    !HostDividable<rund::compute::Fixed<16, 16>, rund::compute::Fixed<17, 15>>);
static_assert(!HostFloatCastable<rund::compute::Fixed<16, 16>>);
static_assert(!HostDoubleCastable<rund::compute::Fixed<32, 32>>);
static_assert(!HostFloatingRawFactory<float>);
static_assert(!HostFloatingRawFactory<double>);
static_assert(!HostFloatingRawFactory<long double>);
static_assert(!HostFloatingRatioFactory<float, std::int64_t>);
static_assert(!HostFloatingRatioFactory<std::int64_t, double>);
static_assert(!HostFloatingRatioFactory<long double, long double>);

} // namespace

int CheckArithmetic() {
  using rund::compute::Fixed;
  const auto half32 =
      Fixed<1, 31>::from_ratio(1, 2, rund::compute::Rounding::NearestEven);
  if (!half32) {
    return half32.exit_code();
  }
  std::array<Fixed<1, 31>, 4> input32{Fixed<1, 31>::zero(), *half32,
                                      Fixed<1, 31>::max(), Fixed<1, 31>::min()};
  auto output32 =
      rund::compute::on(rund::compute::Target::cpu(), input32)
          .map("fixed-1-31",
               rund::compute::capture(
                   [](auto value, auto half) {
                     return rund::compute::quantize<Fixed<1, 31>>(value + half);
                   },
                   *half32))
          .collect();
  if (!output32) {
    return output32.exit_code();
  }
  if (output32->size() != input32.size() || (*output32)[0] != *half32) {
    return 2;
  }

  const auto half64 =
      Fixed<1, 63>::from_ratio(1, 2, rund::compute::Rounding::NearestEven);
  if (!half64) {
    return half64.exit_code();
  }
  std::array<Fixed<1, 63>, 4> input64{Fixed<1, 63>::zero(), *half64,
                                      Fixed<1, 63>::max(), Fixed<1, 63>::min()};
  auto output64 =
      rund::compute::on(rund::compute::Target::cpu(), input64)
          .map("fixed-1-63",
               rund::compute::capture(
                   [](auto value, auto half) {
                     return rund::compute::quantize<Fixed<1, 63>>(value + half);
                   },
                   *half64))
          .collect();
  if (!output64) {
    return output64.exit_code();
  }
  if (output64->size() != input64.size() || (*output64)[0] != *half64) {
    return 2;
  }
  return 0;
}

int CheckPolicyArithmetic() {
  using rund::compute::Fixed;
  using PolicyFixed = Fixed<16, 16>;
  const std::array<PolicyFixed, 5u> policy_input{
      PolicyFixed::min(), PolicyFixed::max(), PolicyFixed::zero(),
      PolicyFixed::from_raw(1), PolicyFixed::from_raw(-1)};
  auto policy_output =
      rund::compute::on(rund::compute::Target::cpu(), policy_input)
          .map("fixed-custom-policy-literal",
               [](auto value) {
                 const auto q = rund::compute::quantize<
                     PolicyFixed, rund::compute::Rounding::Down,
                     rund::compute::Overflow::Wrap,
                     rund::compute::Approximation::Exact>(value + value);
                 return rund::compute::quantize<
                     PolicyFixed, rund::compute::Rounding::Down,
                     rund::compute::Overflow::Wrap,
                     rund::compute::Approximation::Exact>(
                     q + PolicyFixed::from_raw(1));
               })
          .collect();
  const std::vector<PolicyFixed> policy_expected{
      PolicyFixed::from_raw(1), PolicyFixed::from_raw(-1),
      PolicyFixed::from_raw(1), PolicyFixed::from_raw(3),
      PolicyFixed::from_raw(-1)};
  if (!policy_output) {
    return policy_output.exit_code();
  }
  if (*policy_output != policy_expected) {
    return 2;
  }

  const std::array<PolicyFixed, 1u> literal_input{PolicyFixed::zero()};
  auto literal_output =
      rund::compute::on(rund::compute::Target::cpu(), literal_input)
          .map(
              "fixed-format-aware-public-literals",
              [](auto value) {
                return rund::compute::quantize<PolicyFixed>(
                    rund::compute::fixed_one(value) +
                    rund::compute::fixed(rund::compute::FixedOp::Half, value) +
                    rund::compute::fixed(rund::compute::FixedOp::Third, value) +
                    rund::compute::fixed(rund::compute::FixedOp::Quarter,
                                         value));
              })
          .collect();
  constexpr std::int32_t literal_raw = (std::int32_t{1} << 16u) +
                                       (std::int32_t{1} << 15u) + 21845 +
                                       (std::int32_t{1} << 14u);
  if (!literal_output) {
    return literal_output.exit_code();
  }
  if (*literal_output !=
      std::vector<PolicyFixed>{PolicyFixed::from_raw(literal_raw)}) {
    return 2;
  }

  constexpr std::int32_t one_raw = std::int32_t{1} << 16u;
  const std::array<PolicyFixed, 3u> unit_input{
      PolicyFixed::zero(), PolicyFixed::from_raw(one_raw / 2),
      PolicyFixed::from_raw(one_raw * 2)};
  auto saturated_unit =
      rund::compute::on(rund::compute::Target::cpu(), unit_input)
          .map("fixed-format-aware-saturate",
               [](auto value) {
                 return rund::compute::quantize<PolicyFixed>(
                     rund::compute::saturate(value));
               })
          .collect();
  const std::vector<PolicyFixed> saturated_unit_expected{
      PolicyFixed::zero(), PolicyFixed::from_raw(one_raw / 2),
      PolicyFixed::from_raw(one_raw)};
  if (!saturated_unit) {
    return saturated_unit.exit_code();
  }
  if (*saturated_unit != saturated_unit_expected) {
    return 2;
  }

  auto unit_hash =
      rund::compute::on(rund::compute::Target::cpu(), unit_input)
          .map("fixed-format-aware-unit-hash",
               [](auto value) {
                 return rund::compute::quantize<PolicyFixed>(
                     rund::compute::hash(rund::compute::HashOp::Unit, value));
               })
          .collect();
  if (!unit_hash) {
    return unit_hash.exit_code();
  }
  for (const PolicyFixed value : *unit_hash) {
    if ((static_cast<std::uint32_t>(value.raw()) & 0xffff0000u) != 0u) {
      return 2;
    }
  }

  const std::array<PolicyFixed, 3u> window_input{
      PolicyFixed::zero(), PolicyFixed::from_raw(one_raw / 2),
      PolicyFixed::from_raw(one_raw)};
  auto hamming =
      rund::compute::on(rund::compute::Target::cpu(), window_input)
          .map("fixed-format-aware-window-coefficients",
               [](auto value) {
                 return rund::compute::quantize<
                     PolicyFixed, rund::compute::Rounding::NearestEven,
                     rund::compute::Overflow::Saturate,
                     rund::compute::Approximation::Deterministic>(
                     rund::compute::window(rund::compute::WindowOp::Hamming,
                                           value));
               })
          .collect();
  const std::vector<PolicyFixed> hamming_expected{
      PolicyFixed::from_raw(5242), PolicyFixed::from_raw(one_raw),
      PolicyFixed::from_raw(5242)};
  if (!hamming) {
    return hamming.exit_code();
  }
  if (*hamming != hamming_expected) {
    return 2;
  }
  return 0;
}

} // namespace package_compute::fixed_contract
