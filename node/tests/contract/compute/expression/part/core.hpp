#pragma once

#include "../support/record.hpp"
#include "../support/single.hpp"

namespace rund::node::test_contract::expression {
namespace {

template <class T, class Executor>
  requires is_fixed<T>
[[nodiscard]] int check_operators(Executor &executor,
                                  const rund::compute::Backend backend) {
  using namespace rund::compute;
  const T quarter = fixed_value<T>(1, 4);
  const T half = fixed_value<T>(1, 2);
  const T one = fixed_value<T>(1, 1);
  const std::array<T, 6u> input{
      fixed_value<T>(-3, 4), fixed_value<T>(-1, 2), quarter, half, one,
      fixed_value<T>(3, 2)};

  const int operators = check_record_parity(
      executor, backend, input, "public-fixed-operators", [](auto value) {
        using V = typename decltype(value)::Value;
        const auto zero = fixed_zero(value);
        const auto quarter_value = fixed(FixedOp::Quarter, value);
        const auto half_value = fixed(FixedOp::Half, value);
        return record(
            field<Field<0>>(quantize<V>(value + half_value)),
            field<Field<1>>(quantize<V>(value - quarter_value)),
            field<Field<2>>(quantize<V>(value * half_value)),
            field<Field<3>>(
                quantize<V, Rounding::NearestEven, Overflow::Saturate,
                         Approximation::Deterministic>(value / half_value)),
            field<Field<4>>(quantize<V>(-value)),
            field<Field<5>>(quantize<V>(value & half_value)),
            field<Field<6>>(quantize<V>(value | quarter_value)),
            field<Field<7>>(quantize<V>(value ^ half_value)),
            field<Field<8>>(quantize<V>(min(value, half_value))),
            field<Field<9>>(quantize<V>(max(value, quarter_value))),
            field<Field<10>>(quantize<V>(clamp(value, zero, half_value))),
            field<Field<11>>(
                quantize<V>(select(value < half_value, value, half_value))),
            field<Field<12>>(
                quantize<V>(select(value == zero, fixed_max(value), zero))),
            field<Field<13>>(
                quantize<V>(select(value != zero, fixed_max(value), zero))),
            field<Field<14>>(quantize<V>(
                select((value >= zero) &&
                           ((value < half_value) || !(value <= quarter_value)),
                       fixed_max(value), zero))),
            field<Field<15>>(quantize<V>(bit_not(value))));
      });
  if (operators != 0) {
    return operators;
  }
  const int mask_result = check_single_parity(
      executor, backend, input, "public-fixed-mask",
      [](auto value) { return mask(value != fixed_zero(value)); });
  if (mask_result != 0) {
    return 5 + mask_result;
  }
  const std::array<std::uint32_t, 6u> widened_mask_expected{1u, 1u, 1u,
                                                            1u, 1u, 1u};
  const int widened_mask =
      check_single_expected(executor, backend, input, widened_mask_expected,
                            "public-fixed-widened-mask-golden", [](auto value) {
                              const auto squared = value * value;
                              return mask(squared != fixed_zero(squared));
                            });
  if (widened_mask != 0) {
    return 7 + widened_mask;
  }

  const int exact = check_record_parity(
      executor, backend, input, "public-fixed-exact-functions", [](auto value) {
        using V = typename decltype(value)::Value;
        using A = typename alternate_fixed<V>::Type;
        return record(
            field<Field<0>>(quantize<V>(abs(value))),
            field<Field<1>>(quantize<V>(abs_magnitude(value))),
            field<Field<2>>(quantize<V>(sign(value))),
            field<Field<3>>(quantize<V>(neg_positive_fixed(value))),
            field<Field<4>>(quantize<V>(add_sat(value, value))),
            field<Field<5>>(quantize<V>(sub_sat(value, value))),
            field<Field<6>>(quantize<V>(mul_wrap(value, value))),
            field<Field<7>>(quantize<V>(mul_fixed_scaled(value, value))),
            field<Field<8>>(quantize<V>(mul_unsigned_fixed(value, value))),
            field<Field<9>>(quantize<V>(
                mul_add_fixed(value, value, fixed(FixedOp::Quarter, value)))),
            field<Field<10>>(quantize<V>(shl<1u>(value))),
            field<Field<11>>(quantize<V>(shr_logical<1u>(value))),
            field<Field<12>>(quantize<V>(shr_arithmetic<1u>(value))),
            field<Field<13>>(quantize<A, Rounding::TowardZero>(value)),
            field<Field<14>>(
                quantize<A, Rounding::Down, Overflow::Saturate>(value)),
            field<Field<15>>(quantize<A, Rounding::Up, Overflow::Wrap>(value)));
      });
  if (exact != 0) {
    return 10 + exact;
  }

  const int unsigned_saturation =
      check_single_parity(executor, backend, input,
                          "public-fixed-add-sat-unsigned", [](auto value) {
                            using V = typename decltype(value)::Value;
                            return quantize<V>(add_sat_unsigned(value, value));
                          });
  if (unsigned_saturation != 0) {
    return 15 + unsigned_saturation;
  }

  const T unsigned_max = T::from_raw(static_cast<typename T::Raw>(-1));
  const std::array<T, 6u> unsigned_saturation_expected{
      unsigned_max, unsigned_max,         half,
      one,          fixed_value<T>(2, 1), fixed_value<T>(3, 1)};
  const int unsigned_saturation_golden = check_single_expected(
      executor, backend, input, unsigned_saturation_expected,
      "public-fixed-add-sat-unsigned-golden", [](auto value) {
        using V = typename decltype(value)::Value;
        return quantize<V>(add_sat_unsigned(value, value));
      });
  if (unsigned_saturation_golden != 0) {
    return 20 + unsigned_saturation_golden;
  }

  const std::array<T, 4u> positive{quarter, half, one, fixed_value<T>(3, 2)};
  const int nonlinear = check_record_parity(
      executor, backend, positive, "public-fixed-nonlinear-functions",
      [](auto value) {
        using V = typename decltype(value)::Value;
        const auto half_value = fixed(FixedOp::Half, value);
        const auto store = [](auto expression) {
          return quantize<V, Rounding::NearestEven, Overflow::Saturate,
                          Approximation::Deterministic>(expression);
        };
        return record(field<Field<0>>(store(recip(value))),
                      field<Field<1>>(store(sqrt(value))),
                      field<Field<2>>(store(rsqrt(value))),
                      field<Field<3>>(store(sin(value))),
                      field<Field<4>>(store(cos(value))),
                      field<Field<5>>(store(tan(value))),
                      field<Field<6>>(store(exp(value))),
                      field<Field<7>>(store(log(value))),
                      field<Field<8>>(store(value)),
                      field<Field<9>>(store(atan2(value, half_value))),
                      field<Field<10>>(store(value / half_value)),
                      field<Field<11>>(store(div_fixed(value, half_value))),
                      field<Field<12>>(store(mul_fixed(value, half_value))),
                      field<Field<13>>(store(neg(value))),
                      field<Field<14>>(store(fixed(FixedOp::Third, value))),
                      field<Field<15>>(store(fixed_max(value))));
      });
  if (nonlinear != 0) {
    return 20 + nonlinear;
  }
  const int log_product = check_single_parity(
      executor, backend, positive, "public-fixed-power-log-product",
      [](auto value) {
        using V = typename decltype(value)::Value;
        return quantize<V, Rounding::NearestEven, Overflow::Saturate,
                        Approximation::Deterministic>(
            fixed(FixedOp::Half, value) * log(value));
      });
  if (log_product != 0) {
    return 25 + log_product;
  }
  const int power = check_single_parity(
      executor, backend, positive, "public-fixed-power", [](auto value) {
        using V = typename decltype(value)::Value;
        return quantize<V, Rounding::NearestEven, Overflow::Saturate,
                        Approximation::Deterministic>(
            pow(value, fixed(FixedOp::Half, value)));
      });
  if (power != 0) {
    return 30 + power;
  }
  return 0;
}

template <class T, class Executor>
  requires is_fixed<T>
[[nodiscard]] int check_boundaries(Executor &executor,
                                   const rund::compute::Backend backend) {
  using namespace rund::compute;
  using Raw = typename T::Raw;
  constexpr Raw minimum = std::numeric_limits<Raw>::min();
  constexpr Raw maximum = std::numeric_limits<Raw>::max();
  const std::array<T, 5u> input{T::from_raw(minimum), T::from_raw(maximum),
                                T::zero(), T::from_raw(1), T::from_raw(-1)};
  if (const int result = check_record_parity(
          executor, backend, input, "public-fixed-raw-absolute-boundaries",
          [](auto value) {
            using V = typename decltype(value)::Value;
            const auto one_raw =
                select(value == value, V::from_raw(1), V::from_raw(1));
            const auto store = [](auto expression) {
              return quantize<V, Rounding::NearestEven, Overflow::Saturate,
                              Approximation::Deterministic>(expression);
            };
            return record(field<Field<0>>(store(add_sat(value, value))),
                          field<Field<1>>(store(sub_sat(value, one_raw))),
                          field<Field<2>>(store(mul_wrap(value, value))),
                          field<Field<3>>(store(bit_not(value))),
                          field<Field<4>>(store(shl<1u>(value))),
                          field<Field<5>>(store(shr_logical<1u>(value))),
                          field<Field<6>>(store(shr_arithmetic<1u>(value))),
                          field<Field<7>>(store(abs(value))),
                          field<Field<8>>(store(abs_magnitude(value))),
                          field<Field<9>>(store(sign(value))),
                          field<Field<10>>(store(neg_positive_fixed(value))));
          });
      result != 0) {
    return result;
  }
  const std::array<T, 5u> saturated{T::from_raw(minimum), T::from_raw(maximum),
                                    T::zero(), T::from_raw(2), T::from_raw(-2)};
  if (const int result =
          check_single_expected(executor, backend, input, saturated,
                                "public-fixed-raw-add-saturating-golden",
                                [](auto value) {
                                  using V = typename decltype(value)::Value;
                                  return quantize<V>(add_sat(value, value));
                                });
      result != 0) {
    return 10 + result;
  }
  const std::array<T, 5u> magnitude{T::from_raw(maximum), T::from_raw(maximum),
                                    T::zero(), T::from_raw(1), T::from_raw(1)};
  if (const int result = check_single_expected(
          executor, backend, input, magnitude, "public-fixed-raw-abs-golden",
          [](auto value) {
            using V = typename decltype(value)::Value;
            return quantize<V>(abs(value));
          });
      result != 0) {
    return 20 + result;
  }
  const std::array<std::array<T, 5u>, 5u> widened_unary_expected{
      std::array<T, 5u>{T::from_raw(maximum), T::from_raw(maximum), T::zero(),
                        T::from_raw(1), T::from_raw(1)},
      std::array<T, 5u>{T::from_raw(minimum), T::from_raw(maximum), T::zero(),
                        T::from_raw(1), T::from_raw(1)},
      std::array<T, 5u>{T::from_raw(maximum), T::from_raw(maximum), T::zero(),
                        T::from_raw(2), T::from_raw(2)},
      std::array<T, 5u>{T::zero(), T::from_raw(-2), T::zero(), T::from_raw(2),
                        T::from_raw(2)},
      std::array<T, 5u>{T::from_raw(-1), T::from_raw(1), T::zero(),
                        T::from_raw(1), T::from_raw(-1)},
  };
  const auto widened_unary_record = [](auto value) {
    using V = typename decltype(value)::Value;
    const auto direct = abs_magnitude(value);
    const auto widened = abs_magnitude(value + value);
    return record(
        field<Field<0>>(quantize<V, Rounding::NearestEven, Overflow::Saturate,
                                 Approximation::Exact>(direct)),
        field<Field<1>>(quantize<V, Rounding::NearestEven, Overflow::Wrap,
                                 Approximation::Exact>(direct)),
        field<Field<2>>(quantize<V, Rounding::NearestEven, Overflow::Saturate,
                                 Approximation::Exact>(widened)),
        field<Field<3>>(quantize<V, Rounding::NearestEven, Overflow::Wrap,
                                 Approximation::Exact>(widened)),
        field<Field<4>>(quantize<V, Rounding::NearestEven, Overflow::Saturate,
                                 Approximation::Exact>(sign(value + value))));
  };
  if (const int result = check_record_expected(
          executor, backend, input, widened_unary_expected,
          "public-fixed-widened-unary-golden", widened_unary_record);
      result != 0) {
    return 30 + result;
  }
  const std::array<T, 5u> safe{T::from_raw(static_cast<Raw>(minimum + Raw{1})),
                               T::from_raw(maximum), T::zero(), T::from_raw(1),
                               T::from_raw(-1)};
  const std::array<T, 5u> negated{
      T::from_raw(maximum), T::from_raw(static_cast<Raw>(minimum + Raw{1})),
      T::zero(), T::from_raw(-1), T::from_raw(1)};
  if (const int result = check_single_parity(
          executor, backend, safe, "public-fixed-raw-safe-negation",
          [](auto value) {
            using V = typename decltype(value)::Value;
            return quantize<V>(-value);
          });
      result != 0) {
    return 40 + result;
  }
  if (const int result =
          check_single_expected(executor, backend, safe, negated,
                                "public-fixed-raw-safe-negation-golden",
                                [](auto value) {
                                  using V = typename decltype(value)::Value;
                                  return quantize<V>(-value);
                                });
      result != 0) {
    return 50 + result;
  }
  return 0;
}

template <class T, class Executor>
  requires is_fixed<T>
[[nodiscard]] int check_quantize_modes(Executor &executor,
                                       const rund::compute::Backend backend) {
  using namespace rund::compute;
  const std::array<T, 5u> input{T::from_raw(-3), T::from_raw(-1), T::zero(),
                                T::from_raw(1), T::from_raw(3)};
  return check_record_parity(
      executor, backend, input, "public-fixed-quantize-policy-product",
      [](auto value) {
        using V = typename decltype(value)::Value;
        using Target = typename alternate_fixed<V>::Type;
        return record(
            field<Field<0>>(
                quantize<Target, Rounding::TowardZero, Overflow::Saturate,
                         Approximation::Exact>(value)),
            field<Field<1>>(
                quantize<Target, Rounding::TowardZero, Overflow::Saturate,
                         Approximation::Deterministic>(value)),
            field<Field<2>>(
                quantize<Target, Rounding::TowardZero, Overflow::Wrap,
                         Approximation::Exact>(value)),
            field<Field<3>>(
                quantize<Target, Rounding::TowardZero, Overflow::Wrap,
                         Approximation::Deterministic>(value)),
            field<Field<4>>(quantize<Target, Rounding::Down, Overflow::Saturate,
                                     Approximation::Exact>(value)),
            field<Field<5>>(quantize<Target, Rounding::Down, Overflow::Saturate,
                                     Approximation::Deterministic>(value)),
            field<Field<6>>(quantize<Target, Rounding::Down, Overflow::Wrap,
                                     Approximation::Exact>(value)),
            field<Field<7>>(quantize<Target, Rounding::Down, Overflow::Wrap,
                                     Approximation::Deterministic>(value)),
            field<Field<8>>(quantize<Target, Rounding::Up, Overflow::Saturate,
                                     Approximation::Exact>(value)),
            field<Field<9>>(quantize<Target, Rounding::Up, Overflow::Saturate,
                                     Approximation::Deterministic>(value)),
            field<Field<10>>(quantize<Target, Rounding::Up, Overflow::Wrap,
                                      Approximation::Exact>(value)),
            field<Field<11>>(quantize<Target, Rounding::Up, Overflow::Wrap,
                                      Approximation::Deterministic>(value)),
            field<Field<12>>(
                quantize<Target, Rounding::NearestEven, Overflow::Saturate,
                         Approximation::Exact>(value)),
            field<Field<13>>(
                quantize<Target, Rounding::NearestEven, Overflow::Saturate,
                         Approximation::Deterministic>(value)),
            field<Field<14>>(
                quantize<Target, Rounding::NearestEven, Overflow::Wrap,
                         Approximation::Exact>(value)),
            field<Field<15>>(
                quantize<Target, Rounding::NearestEven, Overflow::Wrap,
                         Approximation::Deterministic>(value)));
      });
}

} // namespace
} // namespace rund::node::test_contract::expression
