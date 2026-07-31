#pragma once

#include <rund/compute.hpp>
#include <rund/compute/math.hpp>

#include <array>
#include <cstdio>
#include <tuple>
#include <type_traits>
#include <vector>

namespace package_compute {

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

struct FixedSignedProduct final {};
struct FixedScaledProduct final {};
struct FixedUnsignedProduct final {};
struct FixedFusedProduct final {};

template <class T> inline int DeclaredMultiply() {
  using Raw = typename T::Raw;
  constexpr Raw one = Raw{1} << T::fraction_bits;
  constexpr Raw half = Raw{1} << (T::fraction_bits - 1u);
  const std::array<T, 1u> input{T::from_raw(one)};
  auto output =
      rund::compute::on(rund::compute::Target::cpu(), input)
          .map(
              T::storage_bits == 32u ? "package-fixed-16-16-multiply"
                                     : "package-fixed-20-44-multiply",
              [](auto value) {
                const auto half_value =
                    rund::compute::fixed(rund::compute::FixedOp::Half, value);
                const auto negative_half =
                    rund::compute::neg_positive_fixed(half_value);
                const auto one_value = rund::compute::fixed_one(value);
                const auto negative_one =
                    rund::compute::neg_positive_fixed(one_value);
                return rund::compute::record(
                    rund::compute::field<FixedSignedProduct>(
                        rund::compute::quantize<T>(
                            rund::compute::mul_fixed(value, negative_half))),
                    rund::compute::field<FixedScaledProduct>(
                        rund::compute::quantize<T>(
                            rund::compute::mul_fixed_scaled(value,
                                                            half_value))),
                    rund::compute::field<FixedUnsignedProduct>(
                        rund::compute::quantize<T>(
                            rund::compute::mul_unsigned_fixed(value,
                                                              half_value))),
                    rund::compute::field<FixedFusedProduct>(
                        rund::compute::quantize<T>(rund::compute::mul_add_fixed(
                            value, negative_one, value))));
              })
          .collect();
  if (!output) {
    return output.exit_code();
  }
  if (std::get<0>(*output) != std::vector<T>{T::from_raw(-half)} ||
      std::get<1>(*output) != std::vector<T>{T::from_raw(half)} ||
      std::get<2>(*output) != std::vector<T>{T::from_raw(half)} ||
      std::get<3>(*output) != std::vector<T>{T::zero()}) {
    return 2;
  }
  return 0;
}

template <class T, class Side> inline int MixedPolicyFlow() {
  using namespace rund::compute;
  using Raw = typename T::Raw;
  using SideRaw = typename Side::Raw;
  const Raw one_raw = static_cast<Raw>(Raw{1} << T::fraction_bits);
  const SideRaw side_one_raw =
      static_cast<SideRaw>(SideRaw{1} << Side::fraction_bits);
  const T zero = T::zero();
  const T one = T::from_raw(one_raw);
  const T two = T::from_raw(static_cast<Raw>(one_raw * Raw{2}));
  const Side side_zero = Side::zero();
  const Side side_one = Side::from_raw(side_one_raw);
  const Side side_two =
      Side::from_raw(static_cast<SideRaw>(side_one_raw * SideRaw{2}));
  const std::array<T, 4u> left{zero, one, two, T::from_raw(-one_raw)};
  const std::array<Side, 4u> right{side_one, side_zero,
                                   Side::from_raw(-side_one_raw), side_two};

  auto program =
      on(Target::cpu(2u))
          .template input<T>(left.size())
          .template zip_input<Side>(right.size())
          .branch([](auto first, auto second) {
            const auto normalized =
                zip(first, second)
                    .map("package-fixed-mixed-normalize",
                         [](auto left_value, auto right_value) {
                           const auto stored_left =
                               quantize<T, Rounding::Down, Overflow::Wrap>(
                                   left_value);
                           const auto stored_right =
                               quantize<T, Rounding::Down, Overflow::Wrap>(
                                   right_value);
                           return quantize<T, Rounding::Down, Overflow::Wrap>(
                               stored_left + stored_right);
                         });
            const auto mapped =
                normalized.map("package-fixed-policy-map", [](auto value) {
                  return quantize<T, Rounding::Down, Overflow::Wrap>(value);
                });
            return mapped.combine(
                "package-fixed-policy-combine", normalized,
                [](auto left_value, auto right_value) {
                  return quantize<T, Rounding::Down, Overflow::Wrap>(
                      left_value + right_value);
                });
          })
          .compile();
  if (!program) {
    std::fprintf(stderr, "package mixed Fixed compile: %.*s\n",
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return program.exit_code();
  }
  const auto graph = program->graph();
  if (graph.inputs.size() != 2u || graph.outputs.size() != 1u) {
    return 2;
  }
  const auto &side_input = graph.resources[graph.inputs[1u] - 1u];
  const auto &output = graph.resources[graph.outputs.front() - 1u];
  if (side_input.integer_bits != Side::integer_bits ||
      side_input.fraction_bits != Side::fraction_bits ||
      output.integer_bits != T::integer_bits ||
      output.fraction_bits != T::fraction_bits ||
      output.rounding != Rounding::Down || output.overflow != Overflow::Wrap) {
    return 2;
  }
  auto job = program->resident(left, right);
  if (!job) {
    return job.exit_code();
  }
  const auto executed = job->run();
  if (!executed) {
    return executed.exit_code();
  }
  auto result = job->read();
  if (!result) {
    return result.exit_code();
  }
  return *result == std::vector<T>{two, two, two, two} ? 0 : 2;
}

inline int Fixed() {
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

  using PolicyFixed = Fixed<16, 16>;
  if (const int result = DeclaredMultiply<PolicyFixed>(); result != 0) {
    return result;
  }
  if (const int result = DeclaredMultiply<Fixed<20, 44>>(); result != 0) {
    return result;
  }
  if (const int result = MixedPolicyFlow<Fixed<16, 16>, Fixed<17, 15>>();
      result != 0) {
    return result;
  }
  if (const int result = MixedPolicyFlow<Fixed<20, 44>, Fixed<21, 43>>();
      result != 0) {
    return result;
  }
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

  auto implicit_storage =
      rund::compute::on(rund::compute::Target::cpu(1u))
          .map<Fixed<16, 16>>("implicit-fixed-storage", 1u,
                              [](auto value) { return value + value; })
          .compile();
  if (implicit_storage ||
      implicit_storage.code() != rund::compute::Code::Binding ||
      implicit_storage.error() != "compute_fixed_quantize_required") {
    std::fprintf(stderr,
                 "package implicit Fixed storage: success=%d reason=%.*s\n",
                 implicit_storage ? 1 : 0,
                 static_cast<int>(implicit_storage.error().size()),
                 implicit_storage.error().data());
    return 2;
  }
  auto approximation_downgrade =
      rund::compute::on(rund::compute::Target::cpu(1u))
          .map<Fixed<16, 16>>("fixed-approximation-downgrade", 1u,
                              [](auto value) {
                                return rund::compute::quantize<Fixed<16, 16>>(
                                    rund::compute::sqrt(value));
                              })
          .compile();
  if (approximation_downgrade ||
      approximation_downgrade.code() != rund::compute::Code::Binding ||
      approximation_downgrade.error() !=
          "compute_fixed_approximation_downgrade") {
    return 2;
  }
  auto fixed_policy_mismatch =
      rund::compute::on(rund::compute::Target::cpu(1u))
          .map<Fixed<16, 16>>(
              "fixed-policy-mismatch-rejected", 1u,
              [](auto value) {
                const auto down_wrap =
                    rund::compute::quantize<Fixed<16, 16>,
                                            rund::compute::Rounding::Down,
                                            rund::compute::Overflow::Wrap>(
                        value);
                const auto up_saturate =
                    rund::compute::quantize<Fixed<16, 16>,
                                            rund::compute::Rounding::Up,
                                            rund::compute::Overflow::Saturate>(
                        value);
                return rund::compute::quantize<Fixed<16, 16>>(down_wrap +
                                                              up_saturate);
              })
          .compile();
  if (fixed_policy_mismatch ||
      fixed_policy_mismatch.code() != rund::compute::Code::Binding ||
      fixed_policy_mismatch.error() != "compute_fixed_format_mismatch") {
    return 2;
  }
  auto precision_capacity =
      rund::compute::on(rund::compute::Target::cpu(1u))
          .map<Fixed<32, 32>>("fixed-129-bit-add-rejected", 1u,
                              [](auto value) {
                                return rund::compute::quantize<Fixed<32, 32>>(
                                    value * value + value);
                              })
          .compile();
  if (precision_capacity ||
      precision_capacity.code() != rund::compute::Code::Capacity ||
      precision_capacity.error() != "compute_fixed_precision_capacity") {
    return 2;
  }
  auto select_precision_capacity =
      rund::compute::on(rund::compute::Target::cpu(1u))
          .map<Fixed<1, 63>>(
              "fixed-129-bit-select-rejected", 1u,
              [](auto value) {
                const auto high_fraction = value * value;
                const auto high_integer = (value + value) + value;
                return rund::compute::quantize<Fixed<1, 63>>(
                    rund::compute::select(value == value, high_fraction,
                                          high_integer));
              })
          .compile();
  if (select_precision_capacity ||
      select_precision_capacity.code() != rund::compute::Code::Capacity ||
      select_precision_capacity.error() != "compute_fixed_precision_capacity") {
    return 2;
  }
  auto clamp_precision_capacity =
      rund::compute::on(rund::compute::Target::cpu(1u))
          .map<Fixed<1, 63>>(
              "fixed-129-bit-clamp-rejected", 1u,
              [](auto value) {
                const auto high_fraction = value * value;
                const auto high_integer = (value + value) + value;
                return rund::compute::quantize<Fixed<1, 63>>(
                    rund::compute::clamp(high_integer, high_fraction,
                                         high_integer));
              })
          .compile();
  if (clamp_precision_capacity ||
      clamp_precision_capacity.code() != rund::compute::Code::Capacity ||
      clamp_precision_capacity.error() != "compute_fixed_precision_capacity") {
    return 2;
  }

  using RescaleSource = Fixed<63, 1>;
  using RescaleTarget = Fixed<1, 63>;
  constexpr std::int64_t scaled_one = std::int64_t{1} << 61u;
  const std::array<RescaleSource, 4u> rescale_input{
      RescaleSource::min(), RescaleSource::max(), RescaleSource::from_raw(1),
      RescaleSource::from_raw(-1)};
  auto saturated =
      rund::compute::on(rund::compute::Target::cpu(), rescale_input)
          .map("fixed-left-rescale-saturate",
               [](auto value) {
                 return rund::compute::quantize<
                     RescaleTarget, rund::compute::Rounding::NearestEven,
                     rund::compute::Overflow::Saturate,
                     rund::compute::Approximation::Exact>(value * value);
               })
          .collect();
  const std::vector<RescaleTarget> saturated_expected{
      RescaleTarget::max(), RescaleTarget::max(),
      RescaleTarget::from_raw(scaled_one), RescaleTarget::from_raw(scaled_one)};
  if (!saturated) {
    return saturated.exit_code();
  }
  if (*saturated != saturated_expected) {
    return 2;
  }
  auto wrapped =
      rund::compute::on(rund::compute::Target::cpu(), rescale_input)
          .map("fixed-left-rescale-wrap",
               [](auto value) {
                 return rund::compute::quantize<
                     RescaleTarget, rund::compute::Rounding::NearestEven,
                     rund::compute::Overflow::Wrap,
                     rund::compute::Approximation::Exact>(value * value);
               })
          .collect();
  const std::vector<RescaleTarget> wrapped_expected{
      RescaleTarget::zero(), RescaleTarget::from_raw(scaled_one),
      RescaleTarget::from_raw(scaled_one), RescaleTarget::from_raw(scaled_one)};
  if (!wrapped) {
    return wrapped.exit_code();
  }
  return *wrapped == wrapped_expected ? 0 : 2;
}

} // namespace package_compute
